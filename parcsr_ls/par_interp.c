
/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
*********************************************************************EHEADER*/
#include "headers.h"

/*---------------------------------------------------------------------------
 * hypre_ParAMGBuildInterp
 *--------------------------------------------------------------------------*/

int
hypre_ParAMGBuildInterp( hypre_ParCSRMatrix   *A,
                         int                  *CF_marker,
                         hypre_ParCSRMatrix   *S,
                         int                   debug_flag,
                         double                trunc_factor,
                         hypre_ParCSRMatrix  **P_ptr)
{

   MPI_Comm 	      comm = hypre_ParCSRMatrixComm(A);   
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   double          *A_diag_data = hypre_CSRMatrixData(A_diag);
   int             *A_diag_i = hypre_CSRMatrixI(A_diag);
   int             *A_diag_j = hypre_CSRMatrixJ(A_diag);

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);   
   double          *A_offd_data = hypre_CSRMatrixData(A_offd);
   int             *A_offd_i = hypre_CSRMatrixI(A_offd);
   int             *A_offd_j = hypre_CSRMatrixJ(A_offd);
   int              num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);
   int             *col_map_offd = hypre_ParCSRMatrixColMapOffd(A);

   hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
   int             *S_diag_i = hypre_CSRMatrixI(S_diag);
   int             *S_diag_j = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix *S_offd = hypre_ParCSRMatrixOffd(S);   
   int             *S_offd_i = hypre_CSRMatrixI(S_offd);
   int             *S_offd_j = hypre_CSRMatrixJ(S_offd);

   hypre_ParCSRMatrix *P;
   int		      *col_map_offd_P;

   int             *CF_marker_offd;

   hypre_CSRMatrix *A_ext;
   
   double          *A_ext_data;
   int             *A_ext_i;
   int             *A_ext_j;

   hypre_CSRMatrix    *P_diag;
   hypre_CSRMatrix    *P_offd;   

   double          *P_diag_data;
   int             *P_diag_i;
   int             *P_diag_j;
   double          *P_offd_data;
   int             *P_offd_i;
   int             *P_offd_j;

   int              P_diag_size, P_offd_size;
   
   int             *P_marker, *P_marker_offd;

   int              jj_counter,jj_counter_offd;
   int              jj_begin_row,jj_begin_row_offd;
   int              jj_end_row,jj_end_row_offd;
   
   int              start_indexing = 0; /* start indexing for P_data at 0 */

   int              n_fine = hypre_CSRMatrixNumRows(A_diag);
   int              n_coarse;

   int              strong_f_marker;

   int             *fine_to_coarse;
   int             *fine_to_coarse_offd;
   int              coarse_counter;
   int              num_cpts_local,total_global_cpts;
   int              num_cols_P_offd,my_first_cpt;
   int             *num_cpts_global;

   int              count;
   
   int              i,i1,i2;
   int              j,jj,jj1;
   int              start;
   int              c_num;
   
   double           diagonal;
   double           sum;
   double           distribute;          
   
   double           zero = 0.0;
   double           one  = 1.0;
   
   int              my_id;
   int              num_procs;
   int              num_sends;
   int              index;
   int             *int_buf_data;

   double           max_coef;
   double           row_sum, scale;
   int              next_open,now_checking,num_lost,start_j;
   int              next_open_offd,now_checking_offd,num_lost_offd;

   int col_1 = hypre_ParCSRMatrixFirstRowIndex(A);
   int local_numrows = hypre_CSRMatrixNumRows(A_diag);
   int col_n = col_1 + local_numrows;

   double           wall_time;  /* for debugging instrumentation  */

   MPI_Comm_size(comm, &num_procs);   
   MPI_Comm_rank(comm,&my_id);

   /*-------------------------------------------------------------------
    * Get the CF_marker data for the off-processor columns
    *-------------------------------------------------------------------*/

   if (debug_flag==4) wall_time = time_getWallclockSeconds();

   CF_marker_offd = hypre_CTAlloc(int, num_cols_A_offd);

   if (!comm_pkg)
   {
	hypre_MatvecCommPkgCreate(A);
	comm_pkg = hypre_ParCSRMatrixCommPkg(A); 
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   int_buf_data = hypre_CTAlloc(int, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
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
      printf("Proc = %d     Interp: Comm 1 CF_marker =    %f\n",
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
   
   if (debug_flag==4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      printf("Proc = %d  Interp: Comm 2   Get A_ext =  %f\n",
                    my_id, wall_time);
      fflush(NULL);
   }

   /*-----------------------------------------------------------------------
    * Determine the number of C-pts on each processor, broadcast,
    * the first C-pt on each processor, and the total number of C-pts
    *----------------------------------------------------------------------*/

   if (debug_flag==4) wall_time = time_getWallclockSeconds();

    num_cpts_global = hypre_CTAlloc(int, num_procs+1);
    num_cpts_local = 0;

    for (i = 0; i < n_fine; i++)
    {
       if (CF_marker[i] >= 0) num_cpts_local++;
    }

    MPI_Allgather(&num_cpts_local,1,MPI_INT,num_cpts_global,1,MPI_INT,comm);
    
    if (debug_flag==4)
    {
       wall_time = time_getWallclockSeconds() - wall_time;
       printf("Proc = %d     Interp: Comm 3 Allgather #C = %f\n",
                     my_id, wall_time);
       fflush(NULL);
    }
    if (debug_flag==4) wall_time = time_getWallclockSeconds();

    my_first_cpt = 0;
    for (i = 0; i < my_id; i++)
    {
       my_first_cpt += num_cpts_global[i];
    }
    total_global_cpts = my_first_cpt;
    for (i = my_id; i < num_procs; i++)
    {
       total_global_cpts += num_cpts_global[i];
    }
    num_cpts_global[num_procs] = total_global_cpts;
    for (i = num_procs-1; i >= 0; i--)
    {
       num_cpts_global[i] = num_cpts_global[i+1] - num_cpts_global[i];
    }

   /*-----------------------------------------------------------------------
    *  First Pass: Determine size of P and fill in fine_to_coarse mapping.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Intialize counters and allocate mapping vector.
    *-----------------------------------------------------------------------*/

   coarse_counter = 0;

   fine_to_coarse = hypre_CTAlloc(int, n_fine);
   for (i = 0; i < n_fine; i++) fine_to_coarse[i] = -1;

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;
      
   /*-----------------------------------------------------------------------
    *  Loop over fine grid.
    *-----------------------------------------------------------------------*/
    
   for (i = 0; i < n_fine; i++)
   {
      
      /*--------------------------------------------------------------------
       *  If i is a C-point, interpolation is the identity. Also set up
       *  mapping vector.
       *--------------------------------------------------------------------*/

      if (CF_marker[i] >= 0)
      {
         jj_counter++;
         fine_to_coarse[i] = coarse_counter;
         coarse_counter++;
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
               jj_counter++;
            }
         }

         if (num_procs > 1)
         {
            for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj++)
            {
               i1 = S_offd_j[jj];           
               if (CF_marker_offd[i1] >= 0)
               {
                  jj_counter_offd++;
               }
            }
         }
      }
   }


   /*-----------------------------------------------------------------------
    *  Allocate  arrays.
    *-----------------------------------------------------------------------*/

   n_coarse = coarse_counter;

   P_diag_size = jj_counter;

   P_diag_i    = hypre_CTAlloc(int, n_fine+1);
   P_diag_j    = hypre_CTAlloc(int, P_diag_size);
   P_diag_data = hypre_CTAlloc(double, P_diag_size);

   P_marker = hypre_CTAlloc(int, n_fine);
   P_marker_offd = hypre_CTAlloc(int, num_cols_A_offd);

   P_offd_size = jj_counter_offd;

   P_offd_i    = hypre_CTAlloc(int, n_fine+1);
   P_offd_j    = hypre_CTAlloc(int, P_offd_size);
   P_offd_data = hypre_CTAlloc(double, P_offd_size);

   /*-----------------------------------------------------------------------
    *  Intialize some stuff.
    *-----------------------------------------------------------------------*/

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   for (i = 0; i < n_fine; i++)
   {      
      P_marker[i] = -1;
   }
 
   for (i = 0; i < num_cols_A_offd; i++)
   {      
      P_marker_offd[i] = -1;
   }
  
   strong_f_marker = -2;

   if (debug_flag==4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      printf("Proc = %d     Interp: Internal work 1 =     %f\n",
                    my_id, wall_time);
      fflush(NULL);
   }

   /*-----------------------------------------------------------------------
    *  Send and receive fine_to_coarse info.
    *-----------------------------------------------------------------------*/ 

   if (debug_flag==4) wall_time = time_getWallclockSeconds();

   fine_to_coarse_offd = hypre_CTAlloc(int, num_cols_A_offd); 

   for (i = 0; i < n_fine; i++) fine_to_coarse[i] += my_first_cpt;
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
      printf("Proc = %d     Interp: Comm 4 FineToCoarse = %f\n",
                    my_id, wall_time);
      fflush(NULL);
   }

   if (debug_flag==4) wall_time = time_getWallclockSeconds();

   for (i = 0; i < n_fine; i++) fine_to_coarse[i] -= my_first_cpt;

   /*-----------------------------------------------------------------------
    *  Loop over fine grid points.
    *-----------------------------------------------------------------------*/
    
   for (i = 0; i  < n_fine  ; i ++)
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

            else
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
                  P_offd_j[jj_counter_offd]  = fine_to_coarse_offd[i1];
                  P_offd_data[jj_counter_offd] = zero;
                  jj_counter_offd++;
               }

               /*-----------------------------------------------------------
                * If neighbor i1 is an F-point, mark it as a strong F-point
                * whose connection needs to be distributed.
                *-----------------------------------------------------------*/

               else
               {
                  P_marker_offd[i1] = strong_f_marker;
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

               /* Diagonal block part of row i1 */
               for (jj1 = A_diag_i[i1]; jj1 < A_diag_i[i1+1]; jj1++)
               {
                  i2 = A_diag_j[jj1];
                  if (P_marker[i2] >= jj_begin_row)
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
                     if (P_marker_offd[i2] >= jj_begin_row_offd)
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
                  if (P_marker[i2] >= jj_begin_row)
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
                     if (P_marker_offd[i2] >= jj_begin_row_offd)
                     {
                         P_offd_data[P_marker_offd[i2]]    
                                  += distribute * A_offd_data[jj1]; 
                     }
                  }
               }
               }
               else
               {
               diagonal += A_diag_data[jj];
               }
            }
            
            /*--------------------------------------------------------------
             * Case 3: neighbor i1 weakly influences i, accumulate a_{i,i1}
             * into the diagonal.
             *--------------------------------------------------------------*/

            else
            {
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

                  for (jj1 = A_ext_i[c_num]; jj1 < A_ext_i[c_num+1]; jj1++)
                  {
                     i2 = A_ext_j[jj1];
                                         
                     if (i2 >= col_1 && i2 < col_n)    
                     {                            
                                           /* in the diagonal block */
                        if (P_marker[i2-col_1] >= jj_begin_row)
                        {
                           sum += A_ext_data[jj1];
                        }
                     }
                     else                       
                     {                          
                                           /* in the off_diagonal block  */
                        j = hypre_BinarySearch(col_map_offd,i2,num_cols_A_offd);
                        if (j != -1)
                        { 
                           if (P_marker_offd[j] >= jj_begin_row_offd)
                           {
			      sum += A_ext_data[jj1];
                           }
                        }
			/* for (j = 0; j < num_cols_A_offd; j++)
                        {
                            if (i2 == abs(CF_marker_offd[j]) )
                            { 
                                if (P_marker_offd[j] >= jj_begin_row_offd)
                                {
				   sum += A_ext_data[jj1];
                                }
				break;
                            }
                        } */
 
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

                     if (i2 >= col_1 && i2 < col_n) /* in the diagonal block */           
                     {
                        if (P_marker[i2-col_1] >= jj_begin_row)
                        {
                           P_diag_data[P_marker[i2-col_1]]
                                     += distribute * A_ext_data[jj1];
                        }
                     }
                     else
                     {
                        /* check to see if it is in the off_diagonal block  */
                        j = hypre_BinarySearch(col_map_offd,i2,num_cols_A_offd);
                        if (j != -1)
                        { 
                           if (P_marker_offd[j] >= jj_begin_row_offd)
                                  P_offd_data[P_marker_offd[j]]
                                     += distribute * A_ext_data[jj1];
                        }
                        /* for (j = 0; j < num_cols_A_offd; j++)
                        {
                            if (i2 == abs(CF_marker_offd[j])) 
                            { 
                               if (P_marker_offd[j] >= jj_begin_row_offd)
                                  P_offd_data[P_marker_offd[j]]
                                     += distribute * A_ext_data[jj1];
                               break;
                            }
                        } */
                     }
                  }
                  }
		  else
                  {
                    diagonal += A_offd_data[jj];
                  }
               }
            
               /*-----------------------------------------------------------
                * Case 3: neighbor i1 weakly influences i, accumulate a_{i,i1}
                * into the diagonal.
                *-----------------------------------------------------------*/

               else
               {
                    diagonal += A_offd_data[jj];
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

/*   change !!!!! */
      strong_f_marker--; 

      P_offd_i[i+1] = jj_counter_offd;
   }
   P_diag_i[i] = jj_counter; 

   /* Compress P, removing coefficients smaller than trunc_factor * Max */

   if (trunc_factor != 0.0)
   {
      next_open = 0;
      now_checking = 0;
      num_lost = 0;
      next_open_offd = 0;
      now_checking_offd = 0;
      num_lost_offd = 0;

      for (i = 0; i < n_fine; i++)
      {
       /*  if (CF_marker[i] < 0) */
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
            P_diag_i[i] -= num_lost;
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
            P_offd_i[i] -= num_lost_offd;

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

   	    scale = row_sum/scale;
	    if (scale != 1)
	    {
   	       for (j = P_diag_i[i]; j < (P_diag_i[i+1]-num_lost); j++)
      	          P_diag_data[j] *= scale;
   	       for (j = P_offd_i[i]; j < (P_offd_i[i+1]-num_lost_offd); j++)
      	          P_offd_data[j] *= scale;
	    }
         }
      }
      P_diag_i[n_fine] -= num_lost;
      P_offd_i[n_fine] -= num_lost_offd;
   }

   num_cols_P_offd = 0;
   if (P_offd_size)
   {
      hypre_TFree(P_marker); 
      P_marker = hypre_CTAlloc(int, P_offd_size);

      for (i=0; i < P_offd_size; i++)
	 P_marker[i] = P_offd_j[i];

      qsort0(P_marker, 0, P_offd_size-1);

      num_cols_P_offd = 1;
      index = P_marker[0];
      for (i=1; i < P_offd_size; i++)
      {
	if (P_marker[i] > index)
	{
 	  index = P_marker[i];
 	  P_marker[num_cols_P_offd++] = index;
  	}
      }

      col_map_offd_P = hypre_CTAlloc(int,num_cols_P_offd);

      for (i=0; i < num_cols_P_offd; i++)
         col_map_offd_P[i] = P_marker[i];

      for (i=0; i < P_offd_size; i++)
	P_offd_j[i] = hypre_BinarySearch(col_map_offd_P,
					 P_offd_j[i],
					 num_cols_P_offd);
   }

   P = hypre_ParCSRMatrixCreate(comm, 
                                hypre_ParCSRMatrixGlobalNumRows(A), 
                                total_global_cpts,
                                hypre_ParCSRMatrixColStarts(A),
                                num_cpts_global,
                                num_cols_P_offd, 
                                P_diag_i[n_fine],
                                P_offd_i[n_fine]);

   P_diag = hypre_ParCSRMatrixDiag(P);
   hypre_CSRMatrixData(P_diag) = P_diag_data; 
   hypre_CSRMatrixI(P_diag) = P_diag_i; 
   hypre_CSRMatrixJ(P_diag) = P_diag_j; 
   hypre_ParCSRMatrixOwnsRowStarts(P) = 0; 

   /*-------------------------------------------------------------------
    * The following block was originally in an 
    *
    *           if (num_cols_P_offd)
    *
    * block, which has been eliminated to ensure that the code 
    * runs on one processor.
    *
    *-------------------------------------------------------------------*/

   P_offd = hypre_ParCSRMatrixOffd(P);
   hypre_CSRMatrixI(P_offd) = P_offd_i; 
   if (num_cols_P_offd)
   { 
	hypre_CSRMatrixData(P_offd) = P_offd_data; 
   	hypre_CSRMatrixJ(P_offd) = P_offd_j; 
   	hypre_ParCSRMatrixColMapOffd(P) = col_map_offd_P;
   } 
   hypre_ParCSRMatrixOffd(P) = P_offd;
   hypre_GetCommPkgRTFromCommPkgA(P,A);

   *P_ptr = P;

   hypre_TFree(CF_marker_offd);
   hypre_TFree(int_buf_data);
   hypre_TFree(fine_to_coarse);
   hypre_TFree(fine_to_coarse_offd);
   hypre_TFree(P_marker);  
   hypre_TFree(P_marker_offd);

   if (num_procs > 1) hypre_CSRMatrixDestroy(A_ext);


   return(0);  

}            
          

