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

/*---------------------------------------------------------------------------
 * hypre_BoomerAMGBuildInterp
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGBuildInterp( hypre_ParCSRMatrix   *A,
                         HYPRE_Int                  *CF_marker,
                         hypre_ParCSRMatrix   *S,
                         HYPRE_Int                  *num_cpts_global,
                         HYPRE_Int                   num_functions,
                         HYPRE_Int                  *dof_func,
                         HYPRE_Int                   debug_flag,
                         HYPRE_Real            trunc_factor,
                         HYPRE_Int		       max_elmts,
                         HYPRE_Int 		      *col_offd_S_to_A,
                         hypre_ParCSRMatrix  **P_ptr)
{

   MPI_Comm 	      comm = hypre_ParCSRMatrixComm(A);   
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real      *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int             *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int             *A_diag_j = hypre_CSRMatrixJ(A_diag);

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);   
   HYPRE_Real      *A_offd_data = hypre_CSRMatrixData(A_offd);
   HYPRE_Int             *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int             *A_offd_j = hypre_CSRMatrixJ(A_offd);
   HYPRE_Int              num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_Int             *col_map_offd = hypre_ParCSRMatrixColMapOffd(A);

   hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int             *S_diag_i = hypre_CSRMatrixI(S_diag);
   HYPRE_Int             *S_diag_j = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix *S_offd = hypre_ParCSRMatrixOffd(S);   
   HYPRE_Int             *S_offd_i = hypre_CSRMatrixI(S_offd);
   HYPRE_Int             *S_offd_j = hypre_CSRMatrixJ(S_offd);

   hypre_ParCSRMatrix *P;
   HYPRE_Int		      *col_map_offd_P;

   HYPRE_Int             *CF_marker_offd = NULL;
   HYPRE_Int             *dof_func_offd = NULL;

   hypre_CSRMatrix *A_ext;
   
   HYPRE_Real      *A_ext_data = NULL;
   HYPRE_Int             *A_ext_i = NULL;
   HYPRE_Int             *A_ext_j = NULL;

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

   HYPRE_Int              strong_f_marker;

   HYPRE_Int             *fine_to_coarse;
   HYPRE_Int             *fine_to_coarse_offd;
   HYPRE_Int             *coarse_counter;
   HYPRE_Int              coarse_shift;
   HYPRE_Int              total_global_cpts;
   HYPRE_Int              num_cols_P_offd,my_first_cpt;

   HYPRE_Int              i,i1,i2;
   HYPRE_Int              j,jl,jj,jj1;
   HYPRE_Int              k,kc;
   HYPRE_Int              start;
   HYPRE_Int              sgn;
   HYPRE_Int              c_num;
   
   HYPRE_Real       diagonal;
   HYPRE_Real       sum;
   HYPRE_Real       distribute;          
   
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
   if (num_functions > 1 && num_cols_A_offd)
	dof_func_offd = hypre_CTAlloc(HYPRE_Int, num_cols_A_offd);

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
   if (num_functions > 1)
   {
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
	 start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
	 for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
		int_buf_data[index++] 
		 = dof_func[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
      }
	
      comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data, 
	dof_func_offd);

      hypre_ParCSRCommHandleDestroy(comm_handle);   
   }

   if (debug_flag==4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d     Interp: Comm 1 CF_marker =    %f\n",
                    my_id, wall_time);
      fflush(NULL);
   }

   /*----------------------------------------------------------------------
    * Get the ghost rows of A
    *---------------------------------------------------------------------*/

   if (debug_flag==4) wall_time = time_getWallclockSeconds();

   if (num_procs > 1)
   {
      A_ext      = hypre_ParCSRMatrixExtractBExt(A,A,1);
      A_ext_i    = hypre_CSRMatrixI(A_ext);
      A_ext_j    = hypre_CSRMatrixJ(A_ext);
      A_ext_data = hypre_CSRMatrixData(A_ext);
   }

   index = 0;
   for (i=0; i < num_cols_A_offd; i++)
   {
      for (j=A_ext_i[i]; j < A_ext_i[i+1]; j++)
      {
         k = A_ext_j[j];
         if (k >= col_1 && k < col_n)
         {
            A_ext_j[index] = k - col_1;
            A_ext_data[index++] = A_ext_data[j];
         }
         else
         {
            kc = hypre_BinarySearch(col_map_offd,k,num_cols_A_offd);
            if (kc > -1)
            {
               A_ext_j[index] = -kc-1;
               A_ext_data[index++] = A_ext_data[j];
            }
         }
      }
      A_ext_i[i] = index;
   }
   for (i = num_cols_A_offd; i > 0; i--)
      A_ext_i[i] = A_ext_i[i-1];
   if (num_procs > 1) A_ext_i[0] = 0;
   
   if (debug_flag==4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d  Interp: Comm 2   Get A_ext =  %f\n",
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
       *  If i is an F-point, interpolation is from the C-points that
       *  strongly influence i.
       *--------------------------------------------------------------------*/

      else
      {
         for (jj = S_diag_i[i]; jj < S_diag_i[i+1]; jj++)
         {
            i1 = S_diag_j[jj];           
            if (CF_marker[i1] >= 0)
            {
               jj_count[j]++;
            }
         }

         if (num_procs > 1)
         {
	   if (col_offd_S_to_A)
           {
            for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj++)
            {
               i1 = col_offd_S_to_A[S_offd_j[jj]];           
               if (CF_marker_offd[i1] >= 0)
               {
                  jj_count_offd[j]++;
               }
            }
           }
           else
           {
            for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj++)
            {
               i1 = S_offd_j[jj];           
               if (CF_marker_offd[i1] >= 0)
               {
                  jj_count_offd[j]++;
               }
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
    
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,j,jl,i1,i2,jj,jj1,ns,ne,size,rest,sum,diagonal,distribute,P_marker,P_marker_offd,strong_f_marker,jj_counter,jj_counter_offd,sgn,c_num,jj_begin_row,jj_end_row,jj_begin_row_offd,jj_end_row_offd) HYPRE_SMP_SCHEDULE
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
     strong_f_marker = -2;
 
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

         for (jj = S_diag_i[i]; jj < S_diag_i[i+1]; jj++)
         {
            i1 = S_diag_j[jj];   

            /*--------------------------------------------------------------
             * If neighbor i1 is a C-point, set column number in P_diag_j
             * and initialize interpolation weight to zero.
             *--------------------------------------------------------------*/

            if (CF_marker[i1] >= 0)
            {
               P_marker[i1] = jj_counter;
               P_diag_j[jj_counter]    = fine_to_coarse[i1];
               P_diag_data[jj_counter] = zero;
               jj_counter++;
            }

            /*--------------------------------------------------------------
             * If neighbor i1 is an F-point, mark it as a strong F-point
             * whose connection needs to be distributed.
             *--------------------------------------------------------------*/

            else if (CF_marker[i1] != -3)
            {
               P_marker[i1] = strong_f_marker;
            }            
         }
         jj_end_row = jj_counter;

         /* Off-Diagonal part of P */
         P_offd_i[i] = jj_counter_offd;
         jj_begin_row_offd = jj_counter_offd;


         if (num_procs > 1)
         {
           if (col_offd_S_to_A)
           {
            for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj++)
            {
               i1 = col_offd_S_to_A[S_offd_j[jj]];   

               /*-----------------------------------------------------------
                * If neighbor i1 is a C-point, set column number in P_offd_j
                * and initialize interpolation weight to zero.
                *-----------------------------------------------------------*/

               if (CF_marker_offd[i1] >= 0)
               {
                  P_marker_offd[i1] = jj_counter_offd;
                  /*P_offd_j[jj_counter_offd]  = fine_to_coarse_offd[i1];*/
                  P_offd_j[jj_counter_offd]  = i1;
                  P_offd_data[jj_counter_offd] = zero;
                  jj_counter_offd++;
               }

               /*-----------------------------------------------------------
                * If neighbor i1 is an F-point, mark it as a strong F-point
                * whose connection needs to be distributed.
                *-----------------------------------------------------------*/

               else if (CF_marker_offd[i1] != -3)
               {
                  P_marker_offd[i1] = strong_f_marker;
               }            
            }
           }
           else
           {
            for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj++)
            {
               i1 = S_offd_j[jj];   

               /*-----------------------------------------------------------
                * If neighbor i1 is a C-point, set column number in P_offd_j
                * and initialize interpolation weight to zero.
                *-----------------------------------------------------------*/

               if (CF_marker_offd[i1] >= 0)
               {
                  P_marker_offd[i1] = jj_counter_offd;
                  /*P_offd_j[jj_counter_offd]  = fine_to_coarse_offd[i1];*/
                  P_offd_j[jj_counter_offd]  = i1;
                  P_offd_data[jj_counter_offd] = zero;
                  jj_counter_offd++;
               }

               /*-----------------------------------------------------------
                * If neighbor i1 is an F-point, mark it as a strong F-point
                * whose connection needs to be distributed.
                *-----------------------------------------------------------*/

               else if (CF_marker_offd[i1] != -3)
               {
                  P_marker_offd[i1] = strong_f_marker;
               }            
            }
           }
         }
      
         jj_end_row_offd = jj_counter_offd;
         
         diagonal = A_diag_data[A_diag_i[i]];

     
         /* Loop over ith row of A.  First, the diagonal part of A */

         for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
         {
            i1 = A_diag_j[jj];

            /*--------------------------------------------------------------
             * Case 1: neighbor i1 is a C-point and strongly influences i,
             * accumulate a_{i,i1} into the interpolation weight.
             *--------------------------------------------------------------*/

            if (P_marker[i1] >= jj_begin_row)
            {
               P_diag_data[P_marker[i1]] += A_diag_data[jj];
            }

            /*--------------------------------------------------------------
             * Case 2: neighbor i1 is an F-point and strongly influences i,
             * distribute a_{i,i1} to C-points that strongly infuence i.
             * Note: currently no distribution to the diagonal in this case.
             *--------------------------------------------------------------*/
            
            else if (P_marker[i1] == strong_f_marker)
            {
               sum = zero;
               
               /*-----------------------------------------------------------
                * Loop over row of A for point i1 and calculate the sum
                * of the connections to c-points that strongly influence i.
                *-----------------------------------------------------------*/
	       sgn = 1;
	       if (A_diag_data[A_diag_i[i1]] < 0) sgn = -1;
               /* Diagonal block part of row i1 */
               for (jj1 = A_diag_i[i1]; jj1 < A_diag_i[i1+1]; jj1++)
               {
                  i2 = A_diag_j[jj1];
                  if (P_marker[i2] >= jj_begin_row && 
					(sgn*A_diag_data[jj1]) < 0)
                  {
                     sum += A_diag_data[jj1];
                  }
               }

               /* Off-Diagonal block part of row i1 */ 
               if (num_procs > 1)
               {              
                  for (jj1 = A_offd_i[i1]; jj1 < A_offd_i[i1+1]; jj1++)
                  {
                     i2 = A_offd_j[jj1];
                     if (P_marker_offd[i2] >= jj_begin_row_offd
				&& (sgn*A_offd_data[jj1]) < 0)
                     {
                        sum += A_offd_data[jj1];
                     }
                  }
               } 

               if (sum != 0)
	       {
	       distribute = A_diag_data[jj] / sum;
 
               /*-----------------------------------------------------------
                * Loop over row of A for point i1 and do the distribution.
                *-----------------------------------------------------------*/

               /* Diagonal block part of row i1 */
               for (jj1 = A_diag_i[i1]; jj1 < A_diag_i[i1+1]; jj1++)
               {
                  i2 = A_diag_j[jj1];
                  if (P_marker[i2] >= jj_begin_row 
				&& (sgn*A_diag_data[jj1]) < 0)
                  {
                     P_diag_data[P_marker[i2]]
                                  += distribute * A_diag_data[jj1];
                  }
               }

               /* Off-Diagonal block part of row i1 */
               if (num_procs > 1)
               {
                  for (jj1 = A_offd_i[i1]; jj1 < A_offd_i[i1+1]; jj1++)
                  {
                     i2 = A_offd_j[jj1];
                     if (P_marker_offd[i2] >= jj_begin_row_offd
				&& (sgn*A_offd_data[jj1]) < 0)
                     {
                         P_offd_data[P_marker_offd[i2]]    
                                  += distribute * A_offd_data[jj1]; 
                     }
                  }
               }
               }
               else
               {
		  if (num_functions == 1 || dof_func[i] == dof_func[i1])
                     diagonal += A_diag_data[jj];
               }
            }
            
            /*--------------------------------------------------------------
             * Case 3: neighbor i1 weakly influences i, accumulate a_{i,i1}
             * into the diagonal.
             *--------------------------------------------------------------*/

            else if (CF_marker[i1] != -3)
            {
	       if (num_functions == 1 || dof_func[i] == dof_func[i1])
                  diagonal += A_diag_data[jj];
            } 

         }    
       

          /*----------------------------------------------------------------
           * Still looping over ith row of A. Next, loop over the 
           * off-diagonal part of A 
           *---------------------------------------------------------------*/

         if (num_procs > 1)
         {
            for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
            {
               i1 = A_offd_j[jj];

            /*--------------------------------------------------------------
             * Case 1: neighbor i1 is a C-point and strongly influences i,
             * accumulate a_{i,i1} into the interpolation weight.
             *--------------------------------------------------------------*/

               if (P_marker_offd[i1] >= jj_begin_row_offd)
               {
                  P_offd_data[P_marker_offd[i1]] += A_offd_data[jj];
               }

               /*------------------------------------------------------------
                * Case 2: neighbor i1 is an F-point and strongly influences i,
                * distribute a_{i,i1} to C-points that strongly infuence i.
                * Note: currently no distribution to the diagonal in this case.
                *-----------------------------------------------------------*/
            
               else if (P_marker_offd[i1] == strong_f_marker)
               {
                  sum = zero;
               
               /*---------------------------------------------------------
                * Loop over row of A_ext for point i1 and calculate the sum
                * of the connections to c-points that strongly influence i.
                *---------------------------------------------------------*/

                  /* find row number */
                  c_num = A_offd_j[jj];

		  sgn = 1;
		  if (A_ext_data[A_ext_i[c_num]] < 0) sgn = -1;
                  for (jj1 = A_ext_i[c_num]; jj1 < A_ext_i[c_num+1]; jj1++)
                  {
                     i2 = A_ext_j[jj1];
                                         
                     if (i2 > -1)
                     {                            
                                           /* in the diagonal block */
                        if (P_marker[i2] >= jj_begin_row
				&& (sgn*A_ext_data[jj1]) < 0)
                        {
                           sum += A_ext_data[jj1];
                        }
                     }
                     else                       
                     {                          
                                           /* in the off_diagonal block  */
                        if (P_marker_offd[-i2-1] >= jj_begin_row_offd
				&& (sgn*A_ext_data[jj1]) < 0)
                        {
			   sum += A_ext_data[jj1];
                        }
 
                     }

                  }

                  if (sum != 0)
		  {
		  distribute = A_offd_data[jj] / sum;   
                  /*---------------------------------------------------------
                   * Loop over row of A_ext for point i1 and do 
                   * the distribution.
                   *--------------------------------------------------------*/

                  /* Diagonal block part of row i1 */
                          
                  for (jj1 = A_ext_i[c_num]; jj1 < A_ext_i[c_num+1]; jj1++)
                  {
                     i2 = A_ext_j[jj1];

                     if (i2 > -1) /* in the diagonal block */           
                     {
                        if (P_marker[i2] >= jj_begin_row
				&& (sgn*A_ext_data[jj1]) < 0)
                        {
                           P_diag_data[P_marker[i2]]
                                     += distribute * A_ext_data[jj1];
                        }
                     }
                     else
                     {
                        /* in the off_diagonal block  */
                        if (P_marker_offd[-i2-1] >= jj_begin_row_offd
				&& (sgn*A_ext_data[jj1]) < 0)
                           P_offd_data[P_marker_offd[-i2-1]]
                                     += distribute * A_ext_data[jj1];
                     }
                  }
                  }
		  else
                  {
	             if (num_functions == 1 || dof_func[i] == dof_func_offd[i1])
                        diagonal += A_offd_data[jj];
                  }
               }
            
               /*-----------------------------------------------------------
                * Case 3: neighbor i1 weakly influences i, accumulate a_{i,i1}
                * into the diagonal.
                *-----------------------------------------------------------*/

               else if (CF_marker_offd[i1] != -3)
               {
	          if (num_functions == 1 || dof_func[i] == dof_func_offd[i1])
                     diagonal += A_offd_data[jj];
               } 

            }
         }           

        /*-----------------------------------------------------------------
          * Set interpolation weight by dividing by the diagonal.
          *-----------------------------------------------------------------*/

         if (diagonal == 0.0)
         {
            if (print_level)
               hypre_printf(" Warning! zero diagonal! Proc id %d row %d\n", my_id,i); 
            for (jj = jj_begin_row; jj < jj_end_row; jj++)
            {
               P_diag_data[jj] = 0.0;
            }
            for (jj = jj_begin_row_offd; jj < jj_end_row_offd; jj++)
            {
               P_offd_data[jj] = 0.0;
            }
         }
         else
         {
            for (jj = jj_begin_row; jj < jj_end_row; jj++)
            {
               P_diag_data[jj] /= -diagonal;
            }
            for (jj = jj_begin_row_offd; jj < jj_end_row_offd; jj++)
            {
               P_offd_data[jj] /= -diagonal;
            }
         }
           
      }

      strong_f_marker--; 

      P_offd_i[i+1] = jj_counter_offd;
     }
     hypre_TFree(P_marker);
     hypre_TFree(P_marker_offd);
   }

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

   /* Compress P, removing coefficients smaller than trunc_factor * Max */

   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      hypre_BoomerAMGInterpTruncation(P, trunc_factor, max_elmts);
      P_diag_data = hypre_CSRMatrixData(P_diag);
      P_diag_i = hypre_CSRMatrixI(P_diag);
      P_diag_j = hypre_CSRMatrixJ(P_diag);
      P_offd_data = hypre_CSRMatrixData(P_offd);
      P_offd_i = hypre_CSRMatrixI(P_offd);
      P_offd_j = hypre_CSRMatrixJ(P_offd);
      P_diag_size = P_diag_i[n_fine];
      P_offd_size = P_offd_i[n_fine];
   }

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
   hypre_TFree(dof_func_offd);
   hypre_TFree(int_buf_data);
   hypre_TFree(fine_to_coarse);
   hypre_TFree(fine_to_coarse_offd);
   hypre_TFree(coarse_counter);
   hypre_TFree(jj_count);
   hypre_TFree(jj_count_offd);

   if (num_procs > 1) hypre_CSRMatrixDestroy(A_ext);

   return(0);  

}            
          

/*---------------------------------------------------------------------------
 * hypre_BoomerAMGBuildInterpHE
 * interpolation routine for hyperbolic PDEs
 * treats weak fine connections  like strong fine connections
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGBuildInterpHE( hypre_ParCSRMatrix   *A,
                         HYPRE_Int                  *CF_marker,
                         hypre_ParCSRMatrix   *S,
                         HYPRE_Int                  *num_cpts_global,
                         HYPRE_Int                   num_functions,
                         HYPRE_Int                  *dof_func,
                         HYPRE_Int                   debug_flag,
                         HYPRE_Real            trunc_factor,
                         HYPRE_Int		       max_elmts,
                         HYPRE_Int		      *col_offd_S_to_A,
                         hypre_ParCSRMatrix  **P_ptr)
{

   MPI_Comm 	      comm = hypre_ParCSRMatrixComm(A);   
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real      *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int             *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int             *A_diag_j = hypre_CSRMatrixJ(A_diag);

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);   
   HYPRE_Real      *A_offd_data = hypre_CSRMatrixData(A_offd);
   HYPRE_Int             *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int             *A_offd_j = hypre_CSRMatrixJ(A_offd);
   HYPRE_Int              num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_Int             *col_map_offd = hypre_ParCSRMatrixColMapOffd(A);

   hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int             *S_diag_i = hypre_CSRMatrixI(S_diag);
   HYPRE_Int             *S_diag_j = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix *S_offd = hypre_ParCSRMatrixOffd(S);   
   HYPRE_Int             *S_offd_i = hypre_CSRMatrixI(S_offd);
   HYPRE_Int             *S_offd_j = hypre_CSRMatrixJ(S_offd);

   hypre_ParCSRMatrix *P;
   HYPRE_Int		      *col_map_offd_P;

   HYPRE_Int             *CF_marker_offd = NULL;
   HYPRE_Int             *dof_func_offd = NULL;

   hypre_CSRMatrix *A_ext;
   
   HYPRE_Real      *A_ext_data = NULL;
   HYPRE_Int             *A_ext_i = NULL;
   HYPRE_Int             *A_ext_j = NULL;

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

   HYPRE_Int              i,i1,i2;
   HYPRE_Int              j,jl,jj,jj1;
   HYPRE_Int              k, kc;
   HYPRE_Int              start;
   HYPRE_Int              sgn;
   HYPRE_Int              c_num;
   
   HYPRE_Real       diagonal;
   HYPRE_Real       sum;
   HYPRE_Real       distribute;          
   
   HYPRE_Real       zero = 0.0;
   HYPRE_Real       one  = 1.0;
   
   HYPRE_Int              my_id;
   HYPRE_Int              num_procs;
   HYPRE_Int              num_threads;
   HYPRE_Int              num_sends;
   HYPRE_Int              index;
   HYPRE_Int              ns, ne, size, rest;
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

   if (debug_flag==4) wall_time = time_getWallclockSeconds();

   if (num_cols_A_offd) CF_marker_offd = hypre_CTAlloc(HYPRE_Int, num_cols_A_offd);
   if (num_functions > 1 && num_cols_A_offd)
	dof_func_offd = hypre_CTAlloc(HYPRE_Int, num_cols_A_offd);

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
   if (num_functions > 1)
   {
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
	 start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
	 for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
		int_buf_data[index++] 
		 = dof_func[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
      }
	
      comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data, 
	dof_func_offd);

      hypre_ParCSRCommHandleDestroy(comm_handle);   
   }

   if (debug_flag==4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d     Interp: Comm 1 CF_marker =    %f\n",
                    my_id, wall_time);
      fflush(NULL);
   }

   /*----------------------------------------------------------------------
    * Get the ghost rows of A
    *---------------------------------------------------------------------*/

   if (debug_flag==4) wall_time = time_getWallclockSeconds();

   if (num_procs > 1)
   {
      A_ext      = hypre_ParCSRMatrixExtractBExt(A,A,1);
      A_ext_i    = hypre_CSRMatrixI(A_ext);
      A_ext_j    = hypre_CSRMatrixJ(A_ext);
      A_ext_data = hypre_CSRMatrixData(A_ext);
   }

   index = 0;
   for (i=0; i < num_cols_A_offd; i++)
   {
      for (j=A_ext_i[i]; j < A_ext_i[i+1]; j++)
      {
         k = A_ext_j[j];
         if (k >= col_1 && k < col_n)
         {
            A_ext_j[index] = k - col_1;
            A_ext_data[index++] = A_ext_data[j];
         }
         else
         {
            kc = hypre_BinarySearch(col_map_offd,k,num_cols_A_offd);
            if (kc > -1)
            {
               A_ext_j[index] = -kc-1;
               A_ext_data[index++] = A_ext_data[j];
            }
         }
      }
      A_ext_i[i] = index;
   }
   for (i = num_cols_A_offd; i > 0; i--)
      A_ext_i[i] = A_ext_i[i-1];
   if (num_procs > 1) A_ext_i[0] = 0;
   
   if (debug_flag==4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d  Interp: Comm 2   Get A_ext =  %f\n",
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
       *  If i is an F-point, interpolation is from the C-points that
       *  strongly influence i.
       *--------------------------------------------------------------------*/

      else
      {
         for (jj = S_diag_i[i]; jj < S_diag_i[i+1]; jj++)
         {
            i1 = S_diag_j[jj];           
            if (CF_marker[i1] >= 0)
            {
               jj_count[j]++;
            }
         }

         if (num_procs > 1)
         {
           if (col_offd_S_to_A)
           {
            for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj++)
            {
               i1 = col_offd_S_to_A[S_offd_j[jj]];           
               if (CF_marker_offd[i1] >= 0)
               {
                  jj_count_offd[j]++;
               }
            }
           }
           else
           {
            for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj++)
            {
               i1 = S_offd_j[jj];           
               if (CF_marker_offd[i1] >= 0)
               {
                  jj_count_offd[j]++;
               }
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
    
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,j,jl,i1,i2,jj,jj1,ns,ne,size,rest,sum,diagonal,distribute,P_marker,P_marker_offd,jj_counter,jj_counter_offd,sgn,c_num,jj_begin_row,jj_end_row,jj_begin_row_offd,jj_end_row_offd) HYPRE_SMP_SCHEDULE
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

         for (jj = S_diag_i[i]; jj < S_diag_i[i+1]; jj++)
         {
            i1 = S_diag_j[jj];   

            /*--------------------------------------------------------------
             * If neighbor i1 is a C-point, set column number in P_diag_j
             * and initialize interpolation weight to zero.
             *--------------------------------------------------------------*/

            if (CF_marker[i1] >= 0)
            {
               P_marker[i1] = jj_counter;
               P_diag_j[jj_counter]    = fine_to_coarse[i1];
               P_diag_data[jj_counter] = zero;
               jj_counter++;
            }

         }
         jj_end_row = jj_counter;

         /* Off-Diagonal part of P */
         P_offd_i[i] = jj_counter_offd;
         jj_begin_row_offd = jj_counter_offd;


         if (num_procs > 1)
         {
	   if (col_offd_S_to_A)
           {
            for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj++)
            {
               i1 = col_offd_S_to_A[S_offd_j[jj]];   

               /*-----------------------------------------------------------
                * If neighbor i1 is a C-point, set column number in P_offd_j
                * and initialize interpolation weight to zero.
                *-----------------------------------------------------------*/

               if (CF_marker_offd[i1] >= 0)
               {
                  P_marker_offd[i1] = jj_counter_offd;
                  P_offd_j[jj_counter_offd]  = i1;
                  P_offd_data[jj_counter_offd] = zero;
                  jj_counter_offd++;
               }
            }
           }
           else
           {
            for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj++)
            {
               i1 = S_offd_j[jj];   

               /*-----------------------------------------------------------
                * If neighbor i1 is a C-point, set column number in P_offd_j
                * and initialize interpolation weight to zero.
                *-----------------------------------------------------------*/

               if (CF_marker_offd[i1] >= 0)
               {
                  P_marker_offd[i1] = jj_counter_offd;
                  P_offd_j[jj_counter_offd]  = i1;
                  P_offd_data[jj_counter_offd] = zero;
                  jj_counter_offd++;
               }
            }
           }
         }
      
         jj_end_row_offd = jj_counter_offd;
         
         diagonal = A_diag_data[A_diag_i[i]];

     
         /* Loop over ith row of A.  First, the diagonal part of A */

         for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
         {
            i1 = A_diag_j[jj];

            /*--------------------------------------------------------------
             * Case 1: neighbor i1 is a C-point and strongly influences i,
             * accumulate a_{i,i1} into the interpolation weight.
             *--------------------------------------------------------------*/

            if (P_marker[i1] >= jj_begin_row)
            {
               P_diag_data[P_marker[i1]] += A_diag_data[jj];
            }

            /*--------------------------------------------------------------
             * Case 2: neighbor i1 is an F-point and influences i,
             * distribute a_{i,i1} to C-points that strongly influence i.
             * Note: currently no distribution to the diagonal in this case.
             *--------------------------------------------------------------*/
            
            else 
            {
               sum = zero;
               
               /*-----------------------------------------------------------
                * Loop over row of A for point i1 and calculate the sum
                * of the connections to c-points that strongly influence i.
                *-----------------------------------------------------------*/
	       sgn = 1;
	       if (A_diag_data[A_diag_i[i1]] < 0) sgn = -1;
               /* Diagonal block part of row i1 */
               for (jj1 = A_diag_i[i1]; jj1 < A_diag_i[i1+1]; jj1++)
               {
                  i2 = A_diag_j[jj1];
                  if (P_marker[i2] >= jj_begin_row && 
					(sgn*A_diag_data[jj1]) < 0)
                  {
                     sum += A_diag_data[jj1];
                  }
               }

               /* Off-Diagonal block part of row i1 */ 
               if (num_procs > 1)
               {              
                  for (jj1 = A_offd_i[i1]; jj1 < A_offd_i[i1+1]; jj1++)
                  {
                     i2 = A_offd_j[jj1];
                     if (P_marker_offd[i2] >= jj_begin_row_offd
				&& (sgn*A_offd_data[jj1]) < 0)
                     {
                        sum += A_offd_data[jj1];
                     }
                  }
               } 

               if (sum != 0)
	       {
	       distribute = A_diag_data[jj] / sum;
 
               /*-----------------------------------------------------------
                * Loop over row of A for point i1 and do the distribution.
                *-----------------------------------------------------------*/

               /* Diagonal block part of row i1 */
               for (jj1 = A_diag_i[i1]; jj1 < A_diag_i[i1+1]; jj1++)
               {
                  i2 = A_diag_j[jj1];
                  if (P_marker[i2] >= jj_begin_row 
				&& (sgn*A_diag_data[jj1]) < 0)
                  {
                     P_diag_data[P_marker[i2]]
                                  += distribute * A_diag_data[jj1];
                  }
               }

               /* Off-Diagonal block part of row i1 */
               if (num_procs > 1)
               {
                  for (jj1 = A_offd_i[i1]; jj1 < A_offd_i[i1+1]; jj1++)
                  {
                     i2 = A_offd_j[jj1];
                     if (P_marker_offd[i2] >= jj_begin_row_offd
				&& (sgn*A_offd_data[jj1]) < 0)
                     {
                         P_offd_data[P_marker_offd[i2]]    
                                  += distribute * A_offd_data[jj1]; 
                     }
                  }
               }
               }
               else
               {
		  if (num_functions == 1 || dof_func[i] == dof_func[i1])
                     diagonal += A_diag_data[jj];
               } 
            }
            
         }    
       

          /*----------------------------------------------------------------
           * Still looping over ith row of A. Next, loop over the 
           * off-diagonal part of A 
           *---------------------------------------------------------------*/

         if (num_procs > 1)
         {
            for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
            {
               i1 = A_offd_j[jj];

            /*--------------------------------------------------------------
             * Case 1: neighbor i1 is a C-point and strongly influences i,
             * accumulate a_{i,i1} into the interpolation weight.
             *--------------------------------------------------------------*/

               if (P_marker_offd[i1] >= jj_begin_row_offd)
               {
                  P_offd_data[P_marker_offd[i1]] += A_offd_data[jj];
               }

               /*------------------------------------------------------------
                * Case 2: neighbor i1 is an F-point and influences i,
                * distribute a_{i,i1} to C-points that strongly infuence i.
                * Note: currently no distribution to the diagonal in this case.
                *-----------------------------------------------------------*/
            
               else 
               {
                  sum = zero;
               
               /*---------------------------------------------------------
                * Loop over row of A_ext for point i1 and calculate the sum
                * of the connections to c-points that strongly influence i.
                *---------------------------------------------------------*/

                  /* find row number */
                  c_num = A_offd_j[jj];

		  sgn = 1;
		  if (A_ext_data[A_ext_i[c_num]] < 0) sgn = -1;
                  for (jj1 = A_ext_i[c_num]; jj1 < A_ext_i[c_num+1]; jj1++)
                  {
                     i2 = A_ext_j[jj1];
                                         
                     if (i2 > -1)
                     {                            
                                           /* in the diagonal block */
                        if (P_marker[i2] >= jj_begin_row
				&& (sgn*A_ext_data[jj1]) < 0)
                        {
                           sum += A_ext_data[jj1];
                        }
                     }
                     else                       
                     {                          
                                           /* in the off_diagonal block  */
                        if (P_marker_offd[-i2-1] >= jj_begin_row_offd
				&& (sgn*A_ext_data[jj1]) < 0)
                        {
			      sum += A_ext_data[jj1];
                        }
 
                     }

                  }

                  if (sum != 0)
		  {
		  distribute = A_offd_data[jj] / sum;   
                  /*---------------------------------------------------------
                   * Loop over row of A_ext for point i1 and do 
                   * the distribution.
                   *--------------------------------------------------------*/

                  /* Diagonal block part of row i1 */
                          
                  for (jj1 = A_ext_i[c_num]; jj1 < A_ext_i[c_num+1]; jj1++)
                  {
                     i2 = A_ext_j[jj1];

                     if (i2 > -1) /* in the diagonal block */           
                     {
                        if (P_marker[i2] >= jj_begin_row
				&& (sgn*A_ext_data[jj1]) < 0)
                        {
                           P_diag_data[P_marker[i2]]
                                     += distribute * A_ext_data[jj1];
                        }
                     }
                     else
                     {
                        /* in the off_diagonal block  */
                        if (P_marker_offd[-i2-1] >= jj_begin_row_offd
				&& (sgn*A_ext_data[jj1]) < 0)
                           P_offd_data[P_marker_offd[-i2-1]]
                                     += distribute * A_ext_data[jj1];
                     }
                  }
                  }
		  else
                  {
	             if (num_functions == 1 || dof_func[i] == dof_func_offd[i1])
                        diagonal += A_offd_data[jj];
                  }
               }
            
            }
         }           

        /*-----------------------------------------------------------------
          * Set interpolation weight by dividing by the diagonal.
          *-----------------------------------------------------------------*/

         for (jj = jj_begin_row; jj < jj_end_row; jj++)
         {
            P_diag_data[jj] /= -diagonal;
         }

         for (jj = jj_begin_row_offd; jj < jj_end_row_offd; jj++)
         {
            P_offd_data[jj] /= -diagonal;
         }
           
      }

      P_offd_i[i+1] = jj_counter_offd;
     }
     hypre_TFree(P_marker);
     hypre_TFree(P_marker_offd);
   }
                                                                                
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

   /* Compress P, removing coefficients smaller than trunc_factor * Max */

   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      hypre_BoomerAMGInterpTruncation(P, trunc_factor, max_elmts);
      P_diag_data = hypre_CSRMatrixData(P_diag);
      P_diag_i = hypre_CSRMatrixI(P_diag);
      P_diag_j = hypre_CSRMatrixJ(P_diag);
      P_offd_data = hypre_CSRMatrixData(P_offd);
      P_offd_i = hypre_CSRMatrixI(P_offd);
      P_offd_j = hypre_CSRMatrixJ(P_offd);
      P_diag_size = P_diag_i[n_fine];
      P_offd_size = P_offd_i[n_fine];
   }

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

   hypre_GetCommPkgRTFromCommPkgA(P,A,fine_to_coarse_offd);

   *P_ptr = P;

   hypre_TFree(CF_marker_offd);
   hypre_TFree(dof_func_offd);
   hypre_TFree(int_buf_data);
   hypre_TFree(fine_to_coarse);
   hypre_TFree(fine_to_coarse_offd);
   hypre_TFree(coarse_counter);
   hypre_TFree(jj_count);
   hypre_TFree(jj_count_offd);

   if (num_procs > 1) hypre_CSRMatrixDestroy(A_ext);

   return(0);  

}            
          

/*---------------------------------------------------------------------------
 * hypre_BoomerAMGBuildDirInterp
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGBuildDirInterp( hypre_ParCSRMatrix   *A,
                         HYPRE_Int                  *CF_marker,
                         hypre_ParCSRMatrix   *S,
                         HYPRE_Int                  *num_cpts_global,
                         HYPRE_Int                   num_functions,
                         HYPRE_Int                  *dof_func,
                         HYPRE_Int                   debug_flag,
                         HYPRE_Real            trunc_factor,
                         HYPRE_Int		       max_elmts,
                         HYPRE_Int 		      *col_offd_S_to_A,
                         hypre_ParCSRMatrix  **P_ptr)
{

   MPI_Comm 	      comm = hypre_ParCSRMatrixComm(A);   
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real      *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int             *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int             *A_diag_j = hypre_CSRMatrixJ(A_diag);

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);   
   HYPRE_Real      *A_offd_data = hypre_CSRMatrixData(A_offd);
   HYPRE_Int             *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int             *A_offd_j = hypre_CSRMatrixJ(A_offd);
   HYPRE_Int              num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);

   hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int             *S_diag_i = hypre_CSRMatrixI(S_diag);
   HYPRE_Int             *S_diag_j = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix *S_offd = hypre_ParCSRMatrixOffd(S);   
   HYPRE_Int             *S_offd_i = hypre_CSRMatrixI(S_offd);
   HYPRE_Int             *S_offd_j = hypre_CSRMatrixJ(S_offd);

   hypre_ParCSRMatrix *P;
   HYPRE_Int		      *col_map_offd_P;

   HYPRE_Int             *CF_marker_offd = NULL;
   HYPRE_Int             *dof_func_offd = NULL;

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
   HYPRE_Int              start;
   
   HYPRE_Real       diagonal;
   HYPRE_Real       sum_N_pos, sum_P_pos;
   HYPRE_Real       sum_N_neg, sum_P_neg;
   HYPRE_Real       alfa = 1.0;
   HYPRE_Real       beta = 1.0;
   
   HYPRE_Real       zero = 0.0;
   HYPRE_Real       one  = 1.0;
   
   HYPRE_Int              my_id;
   HYPRE_Int              num_procs;
   HYPRE_Int              num_threads;
   HYPRE_Int              num_sends;
   HYPRE_Int              index;
   HYPRE_Int              ns, ne, size, rest;
   HYPRE_Int             *int_buf_data;

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

   if (debug_flag==4) wall_time = time_getWallclockSeconds();

   if (num_cols_A_offd) CF_marker_offd = hypre_CTAlloc(HYPRE_Int, num_cols_A_offd);
   if (num_functions > 1 && num_cols_A_offd)
	dof_func_offd = hypre_CTAlloc(HYPRE_Int, num_cols_A_offd);

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
   if (num_functions > 1)
   {
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
	 start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
	 for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
		int_buf_data[index++] 
		 = dof_func[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
      }
	
      comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data, 
	dof_func_offd);

      hypre_ParCSRCommHandleDestroy(comm_handle);   
   }

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
       *  If i is an F-point, interpolation is from the C-points that
       *  strongly influence i.
       *--------------------------------------------------------------------*/

      else
      {
         for (jj = S_diag_i[i]; jj < S_diag_i[i+1]; jj++)
         {
            i1 = S_diag_j[jj];           
            if (CF_marker[i1] > 0)
            {
               jj_count[j]++;
            }
         }

         if (num_procs > 1)
         {
	   if (col_offd_S_to_A)
           {
            for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj++)
            {
               i1 = col_offd_S_to_A[S_offd_j[jj]];           
               if (CF_marker_offd[i1] > 0)
               {
                  jj_count_offd[j]++;
               }
            }
           }
           else
           {
            for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj++)
            {
               i1 = S_offd_j[jj];           
               if (CF_marker_offd[i1] > 0)
               {
                  jj_count_offd[j]++;
               }
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
    
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,j,jl,i1,jj,ns,ne,size,rest,diagonal,P_marker,P_marker_offd,jj_counter,jj_counter_offd,jj_begin_row,jj_end_row,jj_begin_row_offd,jj_end_row_offd) HYPRE_SMP_SCHEDULE
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

         for (jj = S_diag_i[i]; jj < S_diag_i[i+1]; jj++)
         {
            i1 = S_diag_j[jj];   

            /*--------------------------------------------------------------
             * If neighbor i1 is a C-point, set column number in P_diag_j
             * and initialize interpolation weight to zero.
             *--------------------------------------------------------------*/

            if (CF_marker[i1] >= 0)
            {
               P_marker[i1] = jj_counter;
               P_diag_j[jj_counter]    = fine_to_coarse[i1];
               P_diag_data[jj_counter] = zero;
               jj_counter++;
            }

         }
         jj_end_row = jj_counter;

         /* Off-Diagonal part of P */
         P_offd_i[i] = jj_counter_offd;
         jj_begin_row_offd = jj_counter_offd;


         if (num_procs > 1)
         {
           if (col_offd_S_to_A)
           {
            for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj++)
            {
               i1 = col_offd_S_to_A[S_offd_j[jj]];   

               /*-----------------------------------------------------------
                * If neighbor i1 is a C-point, set column number in P_offd_j
                * and initialize interpolation weight to zero.
                *-----------------------------------------------------------*/

               if (CF_marker_offd[i1] >= 0)
               {
                  P_marker_offd[i1] = jj_counter_offd;
                  P_offd_j[jj_counter_offd]  = i1;
                  P_offd_data[jj_counter_offd] = zero;
                  jj_counter_offd++;
               }

            }
           }
           else
           {
            for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj++)
            {
               i1 = S_offd_j[jj];   

               /*-----------------------------------------------------------
                * If neighbor i1 is a C-point, set column number in P_offd_j
                * and initialize interpolation weight to zero.
                *-----------------------------------------------------------*/

               if (CF_marker_offd[i1] >= 0)
               {
                  P_marker_offd[i1] = jj_counter_offd;
                  P_offd_j[jj_counter_offd]  = i1;
                  P_offd_data[jj_counter_offd] = zero;
                  jj_counter_offd++;
               }

            }
           }
         }
      
         jj_end_row_offd = jj_counter_offd;
         
         diagonal = A_diag_data[A_diag_i[i]];

     
         /* Loop over ith row of A.  First, the diagonal part of A */
	 sum_N_pos = 0;
	 sum_N_neg = 0;
	 sum_P_pos = 0;
	 sum_P_neg = 0;

         for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
         {
            i1 = A_diag_j[jj];
	    if (num_functions == 1 || dof_func[i1] == dof_func[i])
	    { 
	       if (A_diag_data[jj] > 0)
	          sum_N_pos += A_diag_data[jj];
	       else
	          sum_N_neg += A_diag_data[jj];
	    } 
            /*--------------------------------------------------------------
             * Case 1: neighbor i1 is a C-point and strongly influences i,
             * accumulate a_{i,i1} into the interpolation weight.
             *--------------------------------------------------------------*/

            if (P_marker[i1] >= jj_begin_row)
            {
               P_diag_data[P_marker[i1]] += A_diag_data[jj];
	       if (A_diag_data[jj] > 0)
		  sum_P_pos += A_diag_data[jj];
	       else
		  sum_P_neg += A_diag_data[jj];
            }

         }    
       
          /*----------------------------------------------------------------
           * Still looping over ith row of A. Next, loop over the 
           * off-diagonal part of A 
           *---------------------------------------------------------------*/

         if (num_procs > 1)
         {
            for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
            {
               i1 = A_offd_j[jj];
	       if (num_functions == 1 || dof_func_offd[i1] == dof_func[i])
	       { 
	          if (A_offd_data[jj] > 0)
	             sum_N_pos += A_offd_data[jj];
	          else
	             sum_N_neg += A_offd_data[jj];
	       } 

            /*--------------------------------------------------------------
             * Case 1: neighbor i1 is a C-point and strongly influences i,
             * accumulate a_{i,i1} into the interpolation weight.
             *--------------------------------------------------------------*/

               if (P_marker_offd[i1] >= jj_begin_row_offd)
               {
                  P_offd_data[P_marker_offd[i1]] += A_offd_data[jj];
	          if (A_offd_data[jj] > 0)
		     sum_P_pos += A_offd_data[jj];
	          else
		     sum_P_neg += A_offd_data[jj];
               }

            }
         }           
         if (sum_P_neg) alfa = sum_N_neg/sum_P_neg/diagonal;
         if (sum_P_pos) beta = sum_N_pos/sum_P_pos/diagonal;

        /*-----------------------------------------------------------------
          * Set interpolation weight by dividing by the diagonal.
          *-----------------------------------------------------------------*/

         for (jj = jj_begin_row; jj < jj_end_row; jj++)
         {
            if (P_diag_data[jj]> 0)
               P_diag_data[jj] *= -beta;
            else
               P_diag_data[jj] *= -alfa;
         }

         for (jj = jj_begin_row_offd; jj < jj_end_row_offd; jj++)
         {
            if (P_offd_data[jj]> 0)
               P_offd_data[jj] *= -beta;
            else
               P_offd_data[jj] *= -alfa;
         }
           
      }

      P_offd_i[i+1] = jj_counter_offd;
     }
     hypre_TFree(P_marker);
     hypre_TFree(P_marker_offd);
   }
                                                                                
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

   /* Compress P, removing coefficients smaller than trunc_factor * Max */

   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      hypre_BoomerAMGInterpTruncation(P, trunc_factor, max_elmts);
      P_diag_data = hypre_CSRMatrixData(P_diag);
      P_diag_i = hypre_CSRMatrixI(P_diag);
      P_diag_j = hypre_CSRMatrixJ(P_diag);
      P_offd_data = hypre_CSRMatrixData(P_offd);
      P_offd_i = hypre_CSRMatrixI(P_offd);
      P_offd_j = hypre_CSRMatrixJ(P_offd);
      P_diag_size = P_diag_i[n_fine];
      P_offd_size = P_offd_i[n_fine];
   }

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

   hypre_GetCommPkgRTFromCommPkgA(P,A,fine_to_coarse_offd); 

   *P_ptr = P;

   hypre_TFree(CF_marker_offd);
   hypre_TFree(dof_func_offd);
   hypre_TFree(int_buf_data);
   hypre_TFree(fine_to_coarse);
   hypre_TFree(fine_to_coarse_offd);
   hypre_TFree(coarse_counter);
   hypre_TFree(jj_count);
   hypre_TFree(jj_count_offd);

   return(0);  

}            


HYPRE_Int
hypre_BoomerAMGInterpTruncation( hypre_ParCSRMatrix *P,
                 HYPRE_Real trunc_factor,        
                 HYPRE_Int max_elmts)        
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_INTERP_TRUNC] -= hypre_MPI_Wtime();
#endif

   hypre_CSRMatrix *P_diag = hypre_ParCSRMatrixDiag(P);
   HYPRE_Int *P_diag_i = hypre_CSRMatrixI(P_diag);
   HYPRE_Int *P_diag_j = hypre_CSRMatrixJ(P_diag);
   HYPRE_Real *P_diag_data = hypre_CSRMatrixData(P_diag);
   HYPRE_Int *P_diag_j_new;
   HYPRE_Real *P_diag_data_new;

   hypre_CSRMatrix *P_offd = hypre_ParCSRMatrixOffd(P);
   HYPRE_Int *P_offd_i = hypre_CSRMatrixI(P_offd);
   HYPRE_Int *P_offd_j = hypre_CSRMatrixJ(P_offd);
   HYPRE_Real *P_offd_data = hypre_CSRMatrixData(P_offd);
   HYPRE_Int *P_offd_j_new;
   HYPRE_Real *P_offd_data_new;

   HYPRE_Int n_fine = hypre_CSRMatrixNumRows(P_diag);
   HYPRE_Int num_cols = hypre_CSRMatrixNumCols(P_diag);
   HYPRE_Int i, j, start_j;
   HYPRE_Int ierr = 0;
   HYPRE_Int next_open;
   HYPRE_Int now_checking;
   HYPRE_Int num_lost;
   HYPRE_Int num_lost_global=0;
   HYPRE_Int next_open_offd;
   HYPRE_Int now_checking_offd;
   HYPRE_Int num_lost_offd;
   HYPRE_Int num_lost_global_offd;
   HYPRE_Int P_diag_size;
   HYPRE_Int P_offd_size;
   HYPRE_Int num_elmts;
   HYPRE_Int cnt, cnt_diag, cnt_offd;
   HYPRE_Real max_coef;
   HYPRE_Real row_sum;
   HYPRE_Real scale;
    
   /* Threading variables.  Entry i of num_lost_(offd_)per_thread  holds the
    * number of dropped entries over thread i's row range. Cum_lost_per_thread
    * will temporarily store the cumulative number of dropped entries up to 
    * each thread. */
   HYPRE_Int my_thread_num, num_threads, start, stop;
   HYPRE_Int * max_num_threads = hypre_CTAlloc(HYPRE_Int, 1);
   HYPRE_Int * cum_lost_per_thread;
   HYPRE_Int * num_lost_per_thread;
   HYPRE_Int * num_lost_offd_per_thread;

   /* Initialize threading variables */
   max_num_threads[0] = hypre_NumThreads();
   cum_lost_per_thread = hypre_CTAlloc(HYPRE_Int, max_num_threads[0]);
   num_lost_per_thread = hypre_CTAlloc(HYPRE_Int, max_num_threads[0]);
   num_lost_offd_per_thread = hypre_CTAlloc(HYPRE_Int, max_num_threads[0]);
   for(i=0; i < max_num_threads[0]; i++)
   {
       num_lost_per_thread[i] = 0;
       num_lost_offd_per_thread[i] = 0;
   }

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel private(i,my_thread_num,num_threads,max_coef,j,start_j,row_sum,scale,num_lost,now_checking,next_open,num_lost_offd,now_checking_offd,next_open_offd,start,stop,cnt_diag,cnt_offd,num_elmts,cnt)
#endif
   { 
       my_thread_num = hypre_GetThreadNum();
       num_threads = hypre_NumActiveThreads();
       
       /* Compute each thread's range of rows to truncate and compress.  Note,
        * that i, j and data are all compressed as entries are dropped, but
        * that the compression only occurs locally over each thread's row
        * range.  P_diag_i is only made globally consistent at the end of this
        * routine.  During the dropping phases, P_diag_i[stop] will point to
        * the start of the next thread's row range.  */
       
       /* my row range */
       start = (n_fine/num_threads)*my_thread_num;
       if (my_thread_num == num_threads-1)
       {  stop = n_fine; }
       else
       {  stop = (n_fine/num_threads)*(my_thread_num+1); }


       /* 
        * Truncate based on truncation tolerance 
        */
       if (trunc_factor > 0)
       {
          num_lost = 0;
          num_lost_offd = 0;
          
          next_open = P_diag_i[start];
          now_checking = P_diag_i[start];
          next_open_offd = P_offd_i[start];;
          now_checking_offd = P_offd_i[start];;

          for (i = start; i < stop; i++)
          {
            max_coef = 0;
            for (j = P_diag_i[i]; j < P_diag_i[i+1]; j++)
               max_coef = (max_coef < fabs(P_diag_data[j])) ? 
                      fabs(P_diag_data[j]) : max_coef;
            for (j = P_offd_i[i]; j < P_offd_i[i+1]; j++)
               max_coef = (max_coef < fabs(P_offd_data[j])) ? 
                      fabs(P_offd_data[j]) : max_coef;
            max_coef *= trunc_factor;

            start_j = P_diag_i[i];
            if (num_lost) P_diag_i[i] -= num_lost;  
            row_sum = 0;
            scale = 0;
            for (j = start_j; j < P_diag_i[i+1]; j++)
            {
               row_sum += P_diag_data[now_checking];
               if (fabs(P_diag_data[now_checking]) < max_coef)
               {
                  num_lost++;
                  now_checking++;
               }
               else
               {
                  scale += P_diag_data[now_checking];
                  P_diag_data[next_open] = P_diag_data[now_checking];
                  P_diag_j[next_open] = P_diag_j[now_checking];
                  now_checking++;
                  next_open++;
               }
            }

            start_j = P_offd_i[i];
            if (num_lost_offd) P_offd_i[i] -= num_lost_offd;
            for (j = start_j; j < P_offd_i[i+1]; j++)
            {
               row_sum += P_offd_data[now_checking_offd];
               if (fabs(P_offd_data[now_checking_offd]) < max_coef)
               {
                  num_lost_offd++;
                  now_checking_offd++;
               }
               else
               {
                  scale += P_offd_data[now_checking_offd];
                  P_offd_data[next_open_offd] = P_offd_data[now_checking_offd];
                  P_offd_j[next_open_offd] = P_offd_j[now_checking_offd];
                  now_checking_offd++;
                  next_open_offd++;
               }
            }
            /* normalize row of P */

            if (scale != 0.)
            {
               if (scale != row_sum)
               {
                   scale = row_sum/scale;
                   for (j = P_diag_i[i]; j < (P_diag_i[i+1]-num_lost); j++)
                          P_diag_data[j] *= scale;
                   for (j = P_offd_i[i]; j < (P_offd_i[i+1]-num_lost_offd); j++)
                          P_offd_data[j] *= scale;
               }
            }
          } /* end loop for (i = 0; i < n_fine; i++) */
          
          /* store number of dropped elements and number of threads */
          if(my_thread_num == 0)
          {   max_num_threads[0] = num_threads; }
          num_lost_per_thread[my_thread_num] = num_lost;
          num_lost_offd_per_thread[my_thread_num] = num_lost_offd;
       
       } /* end if (trunc_factor > 0) */
      
       
       /* 
        * Truncate based on capping the nnz per row  
        * 
        */
       if (max_elmts > 0)
       {
           HYPRE_Int P_mxnum, cnt1, last_index, last_index_offd;
           HYPRE_Int *P_aux_j;
           HYPRE_Real *P_aux_data;
           
           /* find maximum row length locally over this row range */
           P_mxnum = 0;
           for (i=start; i<stop; i++)
           {
              /* Note P_diag_i[stop] is the starting point for the next thread 
               * in j and data, not the stop point for this thread */
              last_index = P_diag_i[i+1];
              last_index_offd = P_offd_i[i+1];
              if(i == stop-1)
              {  
                  last_index -= num_lost_per_thread[my_thread_num];   
                  last_index_offd -= num_lost_offd_per_thread[my_thread_num]; 
              }
              cnt1 = last_index-P_diag_i[i] + last_index_offd-P_offd_i[i];
              if (cnt1 > P_mxnum) P_mxnum = cnt1;
           }

           /* Some rows exceed max_elmts, and require truncation.  Essentially,
            * each thread truncates and compresses its range of rows locally. */  
           if (P_mxnum > max_elmts)
           {
                
               num_lost = 0;
               num_lost_offd = 0;

               /* two temporary arrays to hold row i for temporary operations */
               P_aux_j = hypre_CTAlloc(HYPRE_Int, P_mxnum);
               P_aux_data = hypre_CTAlloc(HYPRE_Real, P_mxnum);
               cnt_diag = P_diag_i[start];
               cnt_offd = P_offd_i[start];
               
               for (i = start; i < stop; i++)
               {
                /* Note P_diag_i[stop] is the starting point for the next thread 
                 * in j and data, not the stop point for this thread */
                last_index = P_diag_i[i+1];
                last_index_offd = P_offd_i[i+1];
                if(i == stop-1)
                {  
                    last_index -= num_lost_per_thread[my_thread_num];   
                    last_index_offd -= num_lost_offd_per_thread[my_thread_num]; 
                }  

                row_sum = 0;
                num_elmts = last_index-P_diag_i[i] + last_index_offd-P_offd_i[i];
                if (max_elmts < num_elmts)
                {
                  /* copy both diagonal and off-diag parts of row i to _aux_ arrays */
                  cnt = 0;
                  for (j = P_diag_i[i]; j < last_index; j++)
                  {
                     P_aux_j[cnt] = P_diag_j[j];
                     P_aux_data[cnt++] = P_diag_data[j];
                     row_sum += P_diag_data[j];
                  }
                  num_lost += cnt;
                  cnt1 = cnt;
                  for (j = P_offd_i[i]; j < last_index_offd; j++)
                  {
                     P_aux_j[cnt] = P_offd_j[j]+num_cols;
                     P_aux_data[cnt++] = P_offd_data[j];
                     row_sum += P_offd_data[j];
                  }
                  num_lost_offd += cnt-cnt1;
                  
                  /* sort data */
                  hypre_qsort2abs(P_aux_j,P_aux_data,0,cnt-1);
                  scale = 0;
                  if (i > start)
                  {
                     P_diag_i[i] = cnt_diag;
                     P_offd_i[i] = cnt_offd;
                  }
                  for (j = 0; j < max_elmts; j++)
                  {
                     scale += P_aux_data[j];
                     if (P_aux_j[j] < num_cols)
                     {
                        P_diag_j[cnt_diag] = P_aux_j[j];
                        P_diag_data[cnt_diag++] = P_aux_data[j];
                     }
                     else
                     {
                        P_offd_j[cnt_offd] = P_aux_j[j]-num_cols;
                        P_offd_data[cnt_offd++] = P_aux_data[j];
                     }
                  }
                  num_lost -= cnt_diag-P_diag_i[i];
                  num_lost_offd -= cnt_offd-P_offd_i[i];

                  /* normalize row of P */
                  if (scale != 0.)
                  {
                     if (scale != row_sum)
                     {
                        scale = row_sum/scale;
                        for (j = P_diag_i[i]; j < cnt_diag; j++)
                               P_diag_data[j] *= scale;
                        for (j = P_offd_i[i]; j < cnt_offd; j++)
                               P_offd_data[j] *= scale;
                     }
                  }
                }  /* end if (max_elmts < num_elmts) */

                else
                {
                  /* nothing dropped from this row, but still have to shift entries back
                   * by the number dropped so far */
                  
                  if (P_diag_i[i] != cnt_diag)
                  {
                     start_j = P_diag_i[i];
                     P_diag_i[i] = cnt_diag;
                     for (j = start_j; j < last_index; j++)
                     {
                        P_diag_j[cnt_diag] = P_diag_j[j];
                        P_diag_data[cnt_diag++] = P_diag_data[j];
                     }
                  }
                  else
                     cnt_diag += last_index-P_diag_i[i];
                  
                  if (P_offd_i[i] != cnt_offd)
                  {
                     start_j = P_offd_i[i];
                     P_offd_i[i] = cnt_offd;
                     for (j = start_j; j < last_index_offd; j++)
                     {
                        P_offd_j[cnt_offd] = P_offd_j[j];
                        P_offd_data[cnt_offd++] = P_offd_data[j];
                     }
                  }
                  else
                     cnt_offd += last_index_offd-P_offd_i[i];
                }
               } /* end for (i = 0; i < n_fine; i++) */
               
               num_lost_per_thread[my_thread_num] += num_lost;
               num_lost_offd_per_thread[my_thread_num] += num_lost_offd;
               hypre_TFree(P_aux_j);
               hypre_TFree(P_aux_data);

           } /* end if (P_mxnum > max_elmts) */
       } /* end if (max_elmts > 0) */


       /* Sum up num_lost_global */
#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#endif
       if(my_thread_num == 0)
       {
           num_lost_global = 0;
           num_lost_global_offd = 0;
           for(i = 0; i < max_num_threads[0]; i++)
           {
               num_lost_global += num_lost_per_thread[i]; 
               num_lost_global_offd += num_lost_offd_per_thread[i]; 
           }
       }
#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#endif

       /* 
        * Synchronize and create new diag data structures 
        */
       if (num_lost_global)
       {
          /* Each thread has it's own locally compressed CSR matrix from rows start
           * to stop.  Now, we have to copy each thread's chunk into the new
           * process-wide CSR data structures 
          *
          * First, we compute the new process-wide number of nonzeros (i.e.,
          * P_diag_size), and compute cum_lost_per_thread[k] so that this 
          * entry holds the cumulative sum of entries dropped up to and 
          * including thread k. */
          if(my_thread_num == 0)
          {
              P_diag_size = P_diag_i[n_fine];

              for(i = 0; i < max_num_threads[0]; i++)
              {   
                  P_diag_size -= num_lost_per_thread[i]; 
                  if(i > 0)
                  {   cum_lost_per_thread[i] = num_lost_per_thread[i] + cum_lost_per_thread[i-1]; }
                  else
                  {   cum_lost_per_thread[i] = num_lost_per_thread[i]; }
              }
          
              P_diag_j_new = hypre_CTAlloc(HYPRE_Int,P_diag_size);
              P_diag_data_new = hypre_CTAlloc(HYPRE_Real,P_diag_size);
          }
#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#endif
          
          /* points to next open spot in new data structures for this thread */
          if(my_thread_num == 0)
          {  next_open = 0; }
          else
          {   
              /* remember, cum_lost_per_thread[k] stores the num dropped up to and 
               * including thread k */
              next_open = P_diag_i[start] - cum_lost_per_thread[my_thread_num-1]; 
          }
          /* copy the j and data arrays over */
          for(i = P_diag_i[start]; i < P_diag_i[stop] - num_lost_per_thread[my_thread_num]; i++)
          {
              P_diag_j_new[next_open] = P_diag_j[i];
              P_diag_data_new[next_open] = P_diag_data[i];
              next_open += 1;
          }
          
#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#endif
          /* update P_diag_i with number of dropped entries by all lower ranked
           * threads */
          if(my_thread_num > 0)
          {
              for(i=start; i<stop; i++)
              {
                  P_diag_i[i] -= cum_lost_per_thread[my_thread_num-1];
              }
          }
          
          if(my_thread_num == 0)
          {
              /* Set last entry */
              P_diag_i[n_fine] = P_diag_size ;
              
              hypre_TFree(P_diag_j);
              hypre_TFree(P_diag_data);
              hypre_CSRMatrixJ(P_diag) = P_diag_j_new;
              hypre_CSRMatrixData(P_diag) = P_diag_data_new;
              hypre_CSRMatrixNumNonzeros(P_diag) = P_diag_size;
          }
       }
       

       /* 
        * Synchronize and create new offd data structures 
        */
#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#endif
       if (num_lost_global_offd)
       {
          /* Repeat process for off-diagonal */
          if(my_thread_num == 0)
          {
              P_offd_size = P_offd_i[n_fine];
              for(i = 0; i < max_num_threads[0]; i++)
              {   
                  P_offd_size -= num_lost_offd_per_thread[i]; 
                  if(i > 0)
                  {   cum_lost_per_thread[i] = num_lost_offd_per_thread[i] + cum_lost_per_thread[i-1]; }
                  else
                  {   cum_lost_per_thread[i] = num_lost_offd_per_thread[i]; }
              }
              
              P_offd_j_new = hypre_CTAlloc(HYPRE_Int,P_offd_size);
              P_offd_data_new = hypre_CTAlloc(HYPRE_Real,P_offd_size);
          }
#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#endif
          
          /* points to next open spot in new data structures for this thread */
          if(my_thread_num == 0)
          {  next_open = 0; }
          else
          {   
              /* remember, cum_lost_per_thread[k] stores the num dropped up to and 
               * including thread k */
              next_open = P_offd_i[start] - cum_lost_per_thread[my_thread_num-1]; 
          }

          /* copy the j and data arrays over */
          for(i = P_offd_i[start]; i < P_offd_i[stop] - num_lost_offd_per_thread[my_thread_num]; i++)
          {
              P_offd_j_new[next_open] = P_offd_j[i];
              P_offd_data_new[next_open] = P_offd_data[i];
              next_open += 1;
          }
          
#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#endif
          /* update P_offd_i with number of dropped entries by all lower ranked
           * threads */
          if(my_thread_num > 0)
          {
              for(i=start; i<stop; i++)
              {
                  P_offd_i[i] -= cum_lost_per_thread[my_thread_num-1];
              }
          }

          if(my_thread_num == 0)
          {
              /* Set last entry */
              P_offd_i[n_fine] = P_offd_size ;
              
              hypre_TFree(P_offd_j);
              hypre_TFree(P_offd_data);
              hypre_CSRMatrixJ(P_offd) = P_offd_j_new;
              hypre_CSRMatrixData(P_offd) = P_offd_data_new;
              hypre_CSRMatrixNumNonzeros(P_offd) = P_offd_size;
          }
       }
   
   } /* end parallel region */

   hypre_TFree(max_num_threads);
   hypre_TFree(cum_lost_per_thread);
   hypre_TFree(num_lost_per_thread);
   hypre_TFree(num_lost_offd_per_thread);

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_INTERP_TRUNC] += hypre_MPI_Wtime();
#endif

   return ierr;
}

/* sort both v and w, in place, but based only on entries in w */
void hypre_qsort2abs( HYPRE_Int *v,
             HYPRE_Real *w,
             HYPRE_Int  left,
             HYPRE_Int  right )
{
   HYPRE_Int i, last;
   if (left >= right)
      return;
   hypre_swap2( v, w, left, (left+right)/2);
   last = left;
   for (i = left+1; i <= right; i++)
      if (fabs(w[i]) > fabs(w[left]))
      {
         hypre_swap2(v, w, ++last, i);
      }
   hypre_swap2(v, w, left, last);
   hypre_qsort2abs(v, w, left, last-1);
   hypre_qsort2abs(v, w, last+1, right);
}




/*---------------------------------------------------------------------------
 * hypre_BoomerAMGBuildInterpModUnk - this is a modified interpolation for the unknown approach.  
 * here we need to pass in a strength matrix built on the entire matrix.
 * 
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGBuildInterpModUnk( hypre_ParCSRMatrix   *A,
                                  HYPRE_Int                  *CF_marker,
                                  hypre_ParCSRMatrix   *S,
                                  HYPRE_Int                  *num_cpts_global,
                                  HYPRE_Int                   num_functions,
                                  HYPRE_Int                  *dof_func,
                                  HYPRE_Int                   debug_flag,
                                  HYPRE_Real            trunc_factor,
                                  HYPRE_Int		       max_elmts,
                                  HYPRE_Int 		      *col_offd_S_to_A,
                                  hypre_ParCSRMatrix  **P_ptr)
{

   MPI_Comm 	      comm = hypre_ParCSRMatrixComm(A);   
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real      *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int             *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int             *A_diag_j = hypre_CSRMatrixJ(A_diag);

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);   
   HYPRE_Real      *A_offd_data = hypre_CSRMatrixData(A_offd);
   HYPRE_Int             *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int             *A_offd_j = hypre_CSRMatrixJ(A_offd);
   HYPRE_Int              num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_Int             *col_map_offd = hypre_ParCSRMatrixColMapOffd(A);

   hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int             *S_diag_i = hypre_CSRMatrixI(S_diag);
   HYPRE_Int             *S_diag_j = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix *S_offd = hypre_ParCSRMatrixOffd(S);   
   HYPRE_Int             *S_offd_i = hypre_CSRMatrixI(S_offd);
   HYPRE_Int             *S_offd_j = hypre_CSRMatrixJ(S_offd);

   hypre_ParCSRMatrix *P;
   HYPRE_Int		      *col_map_offd_P;

   HYPRE_Int             *CF_marker_offd = NULL;
   HYPRE_Int             *dof_func_offd = NULL;

   hypre_CSRMatrix *A_ext;
   
   HYPRE_Real      *A_ext_data = NULL;
   HYPRE_Int             *A_ext_i = NULL;
   HYPRE_Int             *A_ext_j = NULL;

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

   HYPRE_Int              strong_f_marker;

   HYPRE_Int             *fine_to_coarse;
   HYPRE_Int             *fine_to_coarse_offd;
   HYPRE_Int             *coarse_counter;
   HYPRE_Int              coarse_shift;
   HYPRE_Int              total_global_cpts;
   HYPRE_Int              num_cols_P_offd,my_first_cpt;

   HYPRE_Int              i,i1,i2;
   HYPRE_Int              j,jl,jj,jj1;
   HYPRE_Int              k,kc;
   HYPRE_Int              start;
   HYPRE_Int              sgn;
   HYPRE_Int              c_num;
   
   HYPRE_Real       diagonal;
   HYPRE_Real       sum;
   HYPRE_Real       distribute;          
   
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
   if (num_functions > 1 && num_cols_A_offd)
	dof_func_offd = hypre_CTAlloc(HYPRE_Int, num_cols_A_offd);

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
   if (num_functions > 1)
   {
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
	 start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
	 for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
		int_buf_data[index++] 
		 = dof_func[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
      }
	
      comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data, 
	dof_func_offd);

      hypre_ParCSRCommHandleDestroy(comm_handle);   
   }

   if (debug_flag==4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d     Interp: Comm 1 CF_marker =    %f\n",
                    my_id, wall_time);
      fflush(NULL);
   }

   /*----------------------------------------------------------------------
    * Get the ghost rows of A
    *---------------------------------------------------------------------*/

   if (debug_flag==4) wall_time = time_getWallclockSeconds();

   if (num_procs > 1)
   {
      A_ext      = hypre_ParCSRMatrixExtractBExt(A,A,1);
      A_ext_i    = hypre_CSRMatrixI(A_ext);
      A_ext_j    = hypre_CSRMatrixJ(A_ext);
      A_ext_data = hypre_CSRMatrixData(A_ext);
   }

   index = 0;
   for (i=0; i < num_cols_A_offd; i++)
   {
      for (j=A_ext_i[i]; j < A_ext_i[i+1]; j++)
      {
         k = A_ext_j[j];
         if (k >= col_1 && k < col_n)
         {
            A_ext_j[index] = k - col_1;
            A_ext_data[index++] = A_ext_data[j];
         }
         else
         {
            kc = hypre_BinarySearch(col_map_offd,k,num_cols_A_offd);
            if (kc > -1)
            {
               A_ext_j[index] = -kc-1;
               A_ext_data[index++] = A_ext_data[j];
            }
         }
      }
      A_ext_i[i] = index;
   }
   for (i = num_cols_A_offd; i > 0; i--)
      A_ext_i[i] = A_ext_i[i-1];
   if (num_procs > 1) A_ext_i[0] = 0;
   
   if (debug_flag==4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d  Interp: Comm 2   Get A_ext =  %f\n",
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
       *  If i is an F-point, interpolation is from the C-points that
       *  strongly influence i.
       *--------------------------------------------------------------------*/

      else
      {
         for (jj = S_diag_i[i]; jj < S_diag_i[i+1]; jj++)
         {
            i1 = S_diag_j[jj];           
            if (CF_marker[i1] >= 0)
            {
               jj_count[j]++;
            }
         }

         if (num_procs > 1)
         {
	   if (col_offd_S_to_A)
           {
            for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj++)
            {
               i1 = col_offd_S_to_A[S_offd_j[jj]];           
               if (CF_marker_offd[i1] >= 0)
               {
                  jj_count_offd[j]++;
               }
            }
           }
           else
           {
            for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj++)
            {
               i1 = S_offd_j[jj];           
               if (CF_marker_offd[i1] >= 0)
               {
                  jj_count_offd[j]++;
               }
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
    
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,j,jl,i1,i2,jj,jj1,ns,ne,size,rest,sum,diagonal,distribute,P_marker,P_marker_offd,strong_f_marker,jj_counter,jj_counter_offd,sgn,c_num,jj_begin_row,jj_end_row,jj_begin_row_offd,jj_end_row_offd) HYPRE_SMP_SCHEDULE
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
     strong_f_marker = -2;
 
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

         for (jj = S_diag_i[i]; jj < S_diag_i[i+1]; jj++)
         {
            i1 = S_diag_j[jj];   

            /*--------------------------------------------------------------
             * If neighbor i1 is a C-point, set column number in P_diag_j
             * and initialize interpolation weight to zero.
             *--------------------------------------------------------------*/

            if (CF_marker[i1] >= 0)
            {
               P_marker[i1] = jj_counter;
               P_diag_j[jj_counter]    = fine_to_coarse[i1];
               P_diag_data[jj_counter] = zero;
               jj_counter++;
            }

            /*--------------------------------------------------------------
             * If neighbor i1 is an F-point, mark it as a strong F-point
             * whose connection needs to be distributed.
             *--------------------------------------------------------------*/

            else if (CF_marker[i1] != -3)
            {
               P_marker[i1] = strong_f_marker;
            }            
         }
         jj_end_row = jj_counter;

         /* Off-Diagonal part of P */
         P_offd_i[i] = jj_counter_offd;
         jj_begin_row_offd = jj_counter_offd;


         if (num_procs > 1)
         {
           if (col_offd_S_to_A)
           {
            for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj++)
            {
               i1 = col_offd_S_to_A[S_offd_j[jj]];   

               /*-----------------------------------------------------------
                * If neighbor i1 is a C-point, set column number in P_offd_j
                * and initialize interpolation weight to zero.
                *-----------------------------------------------------------*/

               if (CF_marker_offd[i1] >= 0)
               {
                  P_marker_offd[i1] = jj_counter_offd;
                  /*P_offd_j[jj_counter_offd]  = fine_to_coarse_offd[i1];*/
                  P_offd_j[jj_counter_offd]  = i1;
                  P_offd_data[jj_counter_offd] = zero;
                  jj_counter_offd++;
               }

               /*-----------------------------------------------------------
                * If neighbor i1 is an F-point, mark it as a strong F-point
                * whose connection needs to be distributed.
                *-----------------------------------------------------------*/

               else if (CF_marker_offd[i1] != -3)
               {
                  P_marker_offd[i1] = strong_f_marker;
               }            
            }
           }
           else
           {
            for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj++)
            {
               i1 = S_offd_j[jj];   

               /*-----------------------------------------------------------
                * If neighbor i1 is a C-point, set column number in P_offd_j
                * and initialize interpolation weight to zero.
                *-----------------------------------------------------------*/

               if (CF_marker_offd[i1] >= 0)
               {
                  P_marker_offd[i1] = jj_counter_offd;
                  /*P_offd_j[jj_counter_offd]  = fine_to_coarse_offd[i1];*/
                  P_offd_j[jj_counter_offd]  = i1;
                  P_offd_data[jj_counter_offd] = zero;
                  jj_counter_offd++;
               }

               /*-----------------------------------------------------------
                * If neighbor i1 is an F-point, mark it as a strong F-point
                * whose connection needs to be distributed.
                *-----------------------------------------------------------*/

               else if (CF_marker_offd[i1] != -3)
               {
                  P_marker_offd[i1] = strong_f_marker;
               }            
            }
           }
         }
      
         jj_end_row_offd = jj_counter_offd;
         
         diagonal = A_diag_data[A_diag_i[i]];

     
         /* Loop over ith row of A.  First, the diagonal part of A */

         for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
         {
            i1 = A_diag_j[jj];

            /*--------------------------------------------------------------
             * Case 1: neighbor i1 is a C-point and strongly influences i,
             * accumulate a_{i,i1} into the interpolation weight.
             *--------------------------------------------------------------*/

            if (P_marker[i1] >= jj_begin_row)
            {
               P_diag_data[P_marker[i1]] += A_diag_data[jj];
            }

            /*--------------------------------------------------------------
             * Case 2: neighbor i1 is an F-point and strongly influences i,
             * distribute a_{i,i1} to C-points that strongly infuence i.
             * Note: currently no distribution to the diagonal in this case.
               
             HERE, we only want to distribut to points of the SAME function type 

             *--------------------------------------------------------------*/
            
            else if (P_marker[i1] == strong_f_marker)
            {
               sum = zero;
               
               /*-----------------------------------------------------------
                * Loop over row of A for point i1 and calculate the sum
                * of the connections to c-points that strongly influence i.
                *-----------------------------------------------------------*/
	       sgn = 1;
	       if (A_diag_data[A_diag_i[i1]] < 0) sgn = -1;
               /* Diagonal block part of row i1 */
               for (jj1 = A_diag_i[i1]; jj1 < A_diag_i[i1+1]; jj1++)
               {
                  i2 = A_diag_j[jj1];
                  if (num_functions == 1 || dof_func[i1] == dof_func[i2])
                  {
                     
                     if (P_marker[i2] >= jj_begin_row && 
                         (sgn*A_diag_data[jj1]) < 0 )
                     {
                        sum += A_diag_data[jj1];
                     }
                  }
                  
               }

               /* Off-Diagonal block part of row i1 */ 
               if (num_procs > 1)
               {              
                  for (jj1 = A_offd_i[i1]; jj1 < A_offd_i[i1+1]; jj1++)
                  {
                     i2 = A_offd_j[jj1];
                     if (num_functions == 1 || dof_func[i1] == dof_func[i2])
                     {
                        if (P_marker_offd[i2] >= jj_begin_row_offd
                            && (sgn*A_offd_data[jj1]) < 0)
                        {
                           sum += A_offd_data[jj1];
                        }
                     }
                  }
               } 

               if (sum != 0)
	       {
                  distribute = A_diag_data[jj] / sum;
 
               /*-----------------------------------------------------------
                * Loop over row of A for point i1 and do the distribution.
                *-----------------------------------------------------------*/

               /* Diagonal block part of row i1 */
                  for (jj1 = A_diag_i[i1]; jj1 < A_diag_i[i1+1]; jj1++)
                  {
                     i2 = A_diag_j[jj1];
                     if (num_functions == 1 || dof_func[i1] == dof_func[i2])
                     {
                        if (P_marker[i2] >= jj_begin_row 
                            && (sgn*A_diag_data[jj1]) < 0)
                        {
                           P_diag_data[P_marker[i2]]
                              += distribute * A_diag_data[jj1];
                        }
                     }
                     
                  }
                  
                  /* Off-Diagonal block part of row i1 */
                  if (num_procs > 1)
                  {
                     for (jj1 = A_offd_i[i1]; jj1 < A_offd_i[i1+1]; jj1++)
                     {
                        i2 = A_offd_j[jj1];
                        if (num_functions == 1 || dof_func[i1] == dof_func[i2])
                        {
                           if (P_marker_offd[i2] >= jj_begin_row_offd
                               && (sgn*A_offd_data[jj1]) < 0)
                           {
                              P_offd_data[P_marker_offd[i2]]    
                                 += distribute * A_offd_data[jj1]; 
                           }
                        }
                     }
                     
                  }
               }
               else /* sum = 0 - only add to diag if the same function type */
               {
		  if (num_functions == 1 || dof_func[i] == dof_func[i1])
                     diagonal += A_diag_data[jj];
               }
            }
            
            /*--------------------------------------------------------------
             * Case 3: neighbor i1 weakly influences i, accumulate a_{i,i1}
             * into the diagonal. (only if the same function type)
             *--------------------------------------------------------------*/

            else if (CF_marker[i1] != -3)
            {
	       if (num_functions == 1 || dof_func[i] == dof_func[i1])
                  diagonal += A_diag_data[jj];
            } 

         }    
       

          /*----------------------------------------------------------------
           * Still looping over ith row of A. Next, loop over the 
           * off-diagonal part of A 
           *---------------------------------------------------------------*/

         if (num_procs > 1)
         {
            for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
            {
               i1 = A_offd_j[jj];

            /*--------------------------------------------------------------
             * Case 1: neighbor i1 is a C-point and strongly influences i,
             * accumulate a_{i,i1} into the interpolation weight.
             *--------------------------------------------------------------*/

               if (P_marker_offd[i1] >= jj_begin_row_offd)
               {
                  P_offd_data[P_marker_offd[i1]] += A_offd_data[jj];
               }

               /*------------------------------------------------------------
                * Case 2: neighbor i1 is an F-point and strongly influences i,
                * distribute a_{i,i1} to C-points that strongly infuence i.
                * Note: currently no distribution to the diagonal in this case.

                AGAIN, we only want to distribut to points of the SAME function type 

                *-----------------------------------------------------------*/
            
               else if (P_marker_offd[i1] == strong_f_marker)
               {
                  sum = zero;
               
               /*---------------------------------------------------------
                * Loop over row of A_ext for point i1 and calculate the sum
                * of the connections to c-points that strongly influence i.
                *---------------------------------------------------------*/

                  /* find row number */
                  c_num = A_offd_j[jj];

		  sgn = 1;
		  if (A_ext_data[A_ext_i[c_num]] < 0) sgn = -1;
                  for (jj1 = A_ext_i[c_num]; jj1 < A_ext_i[c_num+1]; jj1++)
                  {
                     i2 = A_ext_j[jj1];
                     if (num_functions == 1 || dof_func[i1] == dof_func[i2])
                     {                 
                        if (i2 > -1)
                        {                            
                           /* in the diagonal block */
                           if (P_marker[i2] >= jj_begin_row
                               && (sgn*A_ext_data[jj1]) < 0)
                           {
                              sum += A_ext_data[jj1];
                           }
                        }
                        else                       
                        {                          
                           /* in the off_diagonal block  */
                           if (P_marker_offd[-i2-1] >= jj_begin_row_offd
                               && (sgn*A_ext_data[jj1]) < 0)
                           {
                              sum += A_ext_data[jj1];
                           }
                        }
                        
                     }
                  }
                  if (sum != 0)
		  {
                     distribute = A_offd_data[jj] / sum;   
                     /*---------------------------------------------------------
                      * Loop over row of A_ext for point i1 and do 
                      * the distribution.
                      *--------------------------------------------------------*/
                     
                     /* Diagonal block part of row i1 */
                     
                     for (jj1 = A_ext_i[c_num]; jj1 < A_ext_i[c_num+1]; jj1++)
                     {
                        i2 = A_ext_j[jj1];
                        if (num_functions == 1 || dof_func[i1] == dof_func[i2])
                        {
                           if (i2 > -1) /* in the diagonal block */           
                           {
                              if (P_marker[i2] >= jj_begin_row
                                  && (sgn*A_ext_data[jj1]) < 0)
                              {
                                 P_diag_data[P_marker[i2]]
                                    += distribute * A_ext_data[jj1];
                              }
                           }
                           else
                           {
                              /* in the off_diagonal block  */
                              if (P_marker_offd[-i2-1] >= jj_begin_row_offd
                                  && (sgn*A_ext_data[jj1]) < 0)
                                 P_offd_data[P_marker_offd[-i2-1]]
                                    += distribute * A_ext_data[jj1];
                           }
                        }
                     }
                  }
		  else /* sum = 0 */
                  {
	             if (num_functions == 1 || dof_func[i] == dof_func_offd[i1])
                        diagonal += A_offd_data[jj];
                  }
               }
            
               /*-----------------------------------------------------------
                * Case 3: neighbor i1 weakly influences i, accumulate a_{i,i1}
                * into the diagonal.
                *-----------------------------------------------------------*/

               else if (CF_marker_offd[i1] != -3)
               {
	          if (num_functions == 1 || dof_func[i] == dof_func_offd[i1])
                     diagonal += A_offd_data[jj];
               } 

            }
         }           

        /*-----------------------------------------------------------------
          * Set interpolation weight by dividing by the diagonal.
          *-----------------------------------------------------------------*/

         if (diagonal == 0.0)
         {
            if (print_level)
               hypre_printf(" Warning! zero diagonal! Proc id %d row %d\n", my_id,i); 
            for (jj = jj_begin_row; jj < jj_end_row; jj++)
            {
               P_diag_data[jj] = 0.0;
            }
            for (jj = jj_begin_row_offd; jj < jj_end_row_offd; jj++)
            {
               P_offd_data[jj] = 0.0;
            }
         }
         else
         {
            for (jj = jj_begin_row; jj < jj_end_row; jj++)
            {
               P_diag_data[jj] /= -diagonal;
            }
            for (jj = jj_begin_row_offd; jj < jj_end_row_offd; jj++)
            {
               P_offd_data[jj] /= -diagonal;
            }
         }
      }

      strong_f_marker--; 

      P_offd_i[i+1] = jj_counter_offd;
     }
     hypre_TFree(P_marker);
     hypre_TFree(P_marker_offd);
   }

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

   /* Compress P, removing coefficients smaller than trunc_factor * Max */

   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      hypre_BoomerAMGInterpTruncation(P, trunc_factor, max_elmts);
      P_diag_data = hypre_CSRMatrixData(P_diag);
      P_diag_i = hypre_CSRMatrixI(P_diag);
      P_diag_j = hypre_CSRMatrixJ(P_diag);
      P_offd_data = hypre_CSRMatrixData(P_offd);
      P_offd_i = hypre_CSRMatrixI(P_offd);
      P_offd_j = hypre_CSRMatrixJ(P_offd);
      P_diag_size = P_diag_i[n_fine];
      P_offd_size = P_offd_i[n_fine];
   }

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
   hypre_TFree(dof_func_offd);
   hypre_TFree(int_buf_data);
   hypre_TFree(fine_to_coarse);
   hypre_TFree(fine_to_coarse_offd);
   hypre_TFree(coarse_counter);
   hypre_TFree(jj_count);
   hypre_TFree(jj_count_offd);

   if (num_procs > 1) hypre_CSRMatrixDestroy(A_ext);

   return(0);  

}            
         
/*---------------------------------------------------------------------------
 * hypre_BoomerAMGTruncandBuild
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGTruncandBuild( hypre_ParCSRMatrix   *P,
                         HYPRE_Real                trunc_factor,
                         HYPRE_Int                     max_elmts)
{

   hypre_CSRMatrix *P_offd = hypre_ParCSRMatrixOffd(P);
   hypre_ParCSRCommPkg   *commpkg_P = hypre_ParCSRMatrixCommPkg(P);
   HYPRE_Int             *col_map_offd = hypre_ParCSRMatrixColMapOffd(P);
   HYPRE_Int             *P_offd_i = hypre_CSRMatrixI(P_offd);
   HYPRE_Int             *P_offd_j = hypre_CSRMatrixJ(P_offd);
   HYPRE_Int              num_cols_offd = hypre_CSRMatrixNumCols(P_offd);
   HYPRE_Int              n_fine = hypre_CSRMatrixNumRows(P_offd);

   HYPRE_Int             *new_col_map_offd;

   HYPRE_Int              P_offd_size=0, new_num_cols_offd;

   HYPRE_Int             *P_marker;

   HYPRE_Int              i;

   HYPRE_Int              index;

   /* Compress P, removing coefficients smaller than trunc_factor * Max */

   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      hypre_BoomerAMGInterpTruncation(P, trunc_factor, max_elmts);
      P_offd_j = hypre_CSRMatrixJ(P_offd);
      P_offd_i = hypre_CSRMatrixI(P_offd);
      P_offd_size = P_offd_i[n_fine];
   }

   new_num_cols_offd = 0;
   if (P_offd_size)
   {
      P_marker = hypre_CTAlloc(HYPRE_Int, num_cols_offd);

/*#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"*/
      for (i=0; i < num_cols_offd; i++)
         P_marker[i] = 0;

      for (i=0; i < P_offd_size; i++)
      {
         index = P_offd_j[i];
         if (!P_marker[index])
         {
            new_num_cols_offd++;
            P_marker[index] = 1;
         }
      }

      if (new_num_cols_offd)
         new_col_map_offd = hypre_CTAlloc(HYPRE_Int,new_num_cols_offd);

      index = 0;
      for (i=0; i < new_num_cols_offd; i++)
      {
         while (P_marker[index]==0) index++;
         new_col_map_offd[i] = index++;
      }

/*#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"*/
      for (i=0; i < P_offd_size; i++)
        P_offd_j[i] = hypre_BinarySearch(new_col_map_offd,
                                         P_offd_j[i],
                                         new_num_cols_offd);
   }

   index = 0;
   for(i = 0; i < new_num_cols_offd; i++)
   {
       while (P_marker[index] == 0) index++;

       new_col_map_offd[i] = col_map_offd[index];
       index++;
   }

   if (P_offd_size) hypre_TFree(P_marker);

   if (new_num_cols_offd)
   {
        hypre_TFree(col_map_offd);
        hypre_ParCSRMatrixColMapOffd(P) = new_col_map_offd;
        hypre_CSRMatrixNumCols(P_offd) = new_num_cols_offd;
   }

   if (commpkg_P != NULL) hypre_MatvecCommPkgDestroy(commpkg_P);
   hypre_MatvecCommPkgCreate(P);

   return(0);

}
 
hypre_ParCSRMatrix *hypre_CreateC( hypre_ParCSRMatrix  *A,
                                   HYPRE_Real w)
{
   MPI_Comm        comm = hypre_ParCSRMatrixComm(A);

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);

   HYPRE_Real          *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int             *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int             *A_diag_j = hypre_CSRMatrixJ(A_diag);

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);

   HYPRE_Real          *A_offd_data = hypre_CSRMatrixData(A_offd);
   HYPRE_Int             *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int             *A_offd_j = hypre_CSRMatrixJ(A_offd);

   HYPRE_Int *row_starts = hypre_ParCSRMatrixRowStarts(A);
   HYPRE_Int *col_map_offd_A = hypre_ParCSRMatrixColMapOffd(A);
   HYPRE_Int    num_rows = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int    num_cols_offd = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_Int    global_num_rows = hypre_ParCSRMatrixGlobalNumRows(A);

   hypre_ParCSRMatrix *C;
   hypre_CSRMatrix *C_diag;
   hypre_CSRMatrix *C_offd;

   HYPRE_Real          *C_diag_data;
   HYPRE_Int       *C_diag_i;
   HYPRE_Int       *C_diag_j;

   HYPRE_Real          *C_offd_data;
   HYPRE_Int       *C_offd_i;
   HYPRE_Int       *C_offd_j;
   HYPRE_Int       *col_map_offd_C;

   HYPRE_Int i, j, index;
   HYPRE_Real  invdiag;
   HYPRE_Real  w_local = w;

   C = hypre_ParCSRMatrixCreate(comm, global_num_rows, global_num_rows, row_starts,
        row_starts, num_cols_offd, A_diag_i[num_rows], A_offd_i[num_rows]);

   hypre_ParCSRMatrixInitialize(C);

   C_diag = hypre_ParCSRMatrixDiag(C);
   C_offd = hypre_ParCSRMatrixOffd(C);

   C_diag_i = hypre_CSRMatrixI(C_diag);
   C_diag_j = hypre_CSRMatrixJ(C_diag);
   C_diag_data = hypre_CSRMatrixData(C_diag);

   C_offd_i = hypre_CSRMatrixI(C_offd);
   C_offd_j = hypre_CSRMatrixJ(C_offd);
   C_offd_data = hypre_CSRMatrixData(C_offd);

   col_map_offd_C = hypre_ParCSRMatrixColMapOffd(C);

   hypre_ParCSRMatrixOwnsRowStarts(C) = 0;
   hypre_ParCSRMatrixOwnsColStarts(C) = 0;

   for (i=0; i < num_cols_offd; i++)
      col_map_offd_C[i] = col_map_offd_A[i];

   for (i=0; i < num_rows; i++)
   {
      index = A_diag_i[i];
      invdiag = -w/A_diag_data[index];
      C_diag_data[index] = 1.0-w;
      C_diag_j[index] = A_diag_j[index];
      if (w == 0)
      {
         w_local = fabs(A_diag_data[index]);
         for (j = index+1; j < A_diag_i[i+1]; j++)
            w_local += fabs(A_diag_data[j]);
         for (j = A_offd_i[i]; j < A_offd_i[i+1]; j++)
            w_local += fabs(A_offd_data[j]);
         invdiag = -1/w_local;
         C_diag_data[index] = 1.0-A_diag_data[index]/w_local;
      }
      C_diag_i[i] = index;
      C_offd_i[i] = A_offd_i[i];
      for (j = index+1; j < A_diag_i[i+1]; j++)
      {
          C_diag_data[j] = A_diag_data[j]*invdiag;
          C_diag_j[j] = A_diag_j[j];
      }
      for (j = A_offd_i[i]; j < A_offd_i[i+1]; j++)
      {
          C_offd_data[j] = A_offd_data[j]*invdiag;
          C_offd_j[j] = A_offd_j[j];
      }
   }
   C_diag_i[num_rows] = A_diag_i[num_rows];
   C_offd_i[num_rows] = A_offd_i[num_rows];

   return C;
}

