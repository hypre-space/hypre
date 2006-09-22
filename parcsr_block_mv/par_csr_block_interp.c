/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/



#include "headers.h"

/*---------------------------------------------------------------------------
 * hypre_BoomerAMGBlockBuildInterp

   This is the block version of classical R-S interpolation. We use the complete 
   blocks of A (not just the diagonals of these blocks).

   A and P are now Block matrices.  The Strength matrix S is not as it gives
   nodal strengths.

   CF_marker is size number of nodes.

 *--------------------------------------------------------------------------*/

int
hypre_BoomerAMGBuildBlockInterp( hypre_ParCSRBlockMatrix   *A,
                         int                  *CF_marker,
                         hypre_ParCSRMatrix   *S,
                         int                  *num_cpts_global,
                         int                   num_functions,
                         int                  *dof_func,
                         int                   debug_flag,
                         double                trunc_factor,
                         int 		      *col_offd_S_to_A,
                         hypre_ParCSRBlockMatrix  **P_ptr)
{

   MPI_Comm 	      comm = hypre_ParCSRBlockMatrixComm(A);   
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRBlockMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;

   hypre_CSRBlockMatrix *A_diag = hypre_ParCSRBlockMatrixDiag(A);
   double               *A_diag_data = hypre_CSRBlockMatrixData(A_diag);
   int                  *A_diag_i = hypre_CSRBlockMatrixI(A_diag);
   int                  *A_diag_j = hypre_CSRBlockMatrixJ(A_diag);

   int                  block_size = hypre_CSRBlockMatrixBlockSize(A_diag);
   int                  bnnz = block_size*block_size;
   
   hypre_CSRBlockMatrix *A_offd = hypre_ParCSRBlockMatrixOffd(A);   
   double          *A_offd_data = hypre_CSRBlockMatrixData(A_offd);
   int             *A_offd_i = hypre_CSRBlockMatrixI(A_offd);
   int             *A_offd_j = hypre_CSRBlockMatrixJ(A_offd);
   int              num_cols_A_offd = hypre_CSRBlockMatrixNumCols(A_offd);
   int             *col_map_offd = hypre_ParCSRBlockMatrixColMapOffd(A);

   hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
   int             *S_diag_i = hypre_CSRMatrixI(S_diag);
   int             *S_diag_j = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix *S_offd = hypre_ParCSRMatrixOffd(S);   
   int             *S_offd_i = hypre_CSRMatrixI(S_offd);
   int             *S_offd_j = hypre_CSRMatrixJ(S_offd);

   hypre_ParCSRBlockMatrix *P;
   int		           *col_map_offd_P;

   int             *CF_marker_offd;

   hypre_CSRBlockMatrix *A_ext;
   
   double          *A_ext_data;
   int             *A_ext_i;
   int             *A_ext_j;

   hypre_CSRBlockMatrix    *P_diag;
   hypre_CSRBlockMatrix    *P_offd;   

   double          *P_diag_data;
   int             *P_diag_i;
   int             *P_diag_j;
   double          *P_offd_data;
   int             *P_offd_i;
   int             *P_offd_j;

   int              P_diag_size, P_offd_size;
   
   int             *P_marker, *P_marker_offd;

   int              jj_counter,jj_counter_offd;
   int             *jj_count, *jj_count_offd;
   int              jj_begin_row,jj_begin_row_offd;
   int              jj_end_row,jj_end_row_offd;
   
   int              start_indexing = 0; /* start indexing for P_data at 0 */

   int              n_fine = hypre_CSRBlockMatrixNumRows(A_diag);

   int              strong_f_marker;

   int             *fine_to_coarse;
   int             *fine_to_coarse_offd;
   int             *coarse_counter;
   int              coarse_shift;
   int              total_global_cpts;
   int              num_cols_P_offd, my_first_cpt;

   int              bd;
   
   int              i,i1,i2;
   int              j,jl,jj,jj1;
   int              k,kc;
   int              start;

   int              c_num;
   
   int              my_id;
   int              num_procs;
   int              num_threads;
   int              num_sends;
   int              index;
   int              ns, ne, size, rest;
   int             *int_buf_data;

   int col_1 = hypre_ParCSRBlockMatrixFirstRowIndex(A);
   int local_numrows = hypre_CSRBlockMatrixNumRows(A_diag);
   int col_n = col_1 + local_numrows;

   double           wall_time;  /* for debugging instrumentation  */

   
   double           *identity_block;
   double           *zero_block;
   double           *diagonal_block;
   double           *sum_block;
   double           *distribute_block;
   
   int               *P_row_starts, *A_col_starts;
   

   MPI_Comm_size(comm, &num_procs);   
   MPI_Comm_rank(comm,&my_id);
   num_threads = hypre_NumThreads();


#ifdef HYPRE_NO_GLOBAL_PARTITION
   my_first_cpt = num_cpts_global[0];
   if (my_id == (num_procs -1)) total_global_cpts = num_cpts_global[1];
   MPI_Bcast(&total_global_cpts, 1, MPI_INT, num_procs-1, comm);
#else
   my_first_cpt = num_cpts_global[my_id];
   total_global_cpts = num_cpts_global[num_procs];
#endif

   /*-------------------------------------------------------------------
    * Get the CF_marker data for the off-processor columns
    *-------------------------------------------------------------------*/

   if (debug_flag==4) wall_time = time_getWallclockSeconds();

   CF_marker_offd = hypre_CTAlloc(int, num_cols_A_offd);


   if (!comm_pkg)
   {
#ifdef HYPRE_NO_GLOBAL_PARTITION
        hypre_BlockNewCommPkgCreate(A); 
#else
	hypre_BlockMatvecCommPkgCreate(A);
#endif
	comm_pkg = hypre_ParCSRBlockMatrixCommPkg(A); 
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   int_buf_data = hypre_CTAlloc(int, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
						num_sends));

   index = 0;
   for (i = 0; i < num_sends; i++)
   {
	start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
	for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
        {
           int_buf_data[index++] 
                   = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
        }
        
   }
	
   /* we do not need the block version of comm handle - because
      CF_marker corresponds to the nodal matrix.  This call populates
      CF_marker_offd */
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
      A_ext      = hypre_ParCSRBlockMatrixExtractBExt(A, A, 1);
      A_ext_i    = hypre_CSRBlockMatrixI(A_ext);
      A_ext_j    = hypre_CSRBlockMatrixJ(A_ext);
      A_ext_data = hypre_CSRBlockMatrixData(A_ext);
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
            /* for the data field we must get all of the block data */    
            for (bd = 0; bd < bnnz; bd++)
            {
               A_ext_data[index*bnnz + bd] = A_ext_data[j*bnnz + bd];
            }
            index++;
         }
         else
         {
            kc = hypre_BinarySearch(col_map_offd, k, num_cols_A_offd);
            if (kc > -1)
            {
               A_ext_j[index] = -kc-1;
               for (bd = 0; bd < bnnz; bd++)
               {
                  A_ext_data[index*bnnz + bd] = A_ext_data[j*bnnz + bd];
               }
               index++;
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
      printf("Proc = %d  Interp: Comm 2   Get A_ext =  %f\n",
                    my_id, wall_time);
      fflush(NULL);
   }


   /*-----------------------------------------------------------------------
    *  First Pass: Determine size of P and fill in fine_to_coarse mapping.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Intialize counters and allocate mapping vector.
    *-----------------------------------------------------------------------*/

   coarse_counter = hypre_CTAlloc(int, num_threads);
   jj_count = hypre_CTAlloc(int, num_threads);
   jj_count_offd = hypre_CTAlloc(int, num_threads);

   fine_to_coarse = hypre_CTAlloc(int, n_fine);
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
   for (i = 0; i < n_fine; i++) fine_to_coarse[i] = -1;

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;
      
   /*-----------------------------------------------------------------------
    *  Loop over fine grid.
    *-----------------------------------------------------------------------*/

/* RDF: this looks a little tricky, but doable */
#define HYPRE_SMP_PRIVATE i,j,i1,jj,ns,ne,size,rest
#include "../utilities/hypre_smp_forloop.h"
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


     /* loop over the fine grid points */   
     for (i = ns; i < ne; i++)
     {
      
      /*--------------------------------------------------------------------
       *  If i is a C-point, interpolation is the identity. Also set up
       *  mapping vector (fine_to_coarse is the mapping vector).
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

   P_diag_i    = hypre_CTAlloc(int, n_fine+1);
   P_diag_j    = hypre_CTAlloc(int, P_diag_size);
   /* we need to include the size of the blocks in the data size */
   P_diag_data = hypre_CTAlloc(double, P_diag_size*bnnz);

   P_diag_i[n_fine] = jj_counter; 


   P_offd_size = jj_counter_offd;

   P_offd_i    = hypre_CTAlloc(int, n_fine+1);
   P_offd_j    = hypre_CTAlloc(int, P_offd_size);
   /* we need to include the size of the blocks in the data size */
   P_offd_data = hypre_CTAlloc(double, P_offd_size*bnnz);

   /*-----------------------------------------------------------------------
    *  Intialize some stuff.
    *-----------------------------------------------------------------------*/

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   if (debug_flag==4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      printf("Proc = %d     Interp: Internal work 1 =     %f\n",
                    my_id, wall_time);
      fflush(NULL);
   }

   /* we need a block identity and a block of zeros*/
   identity_block = hypre_CTAlloc(double, bnnz);
   zero_block =  hypre_CTAlloc(double, bnnz);
 
   for(i = 0; i < block_size; i++) 
   {
      identity_block[i*block_size + i] = 1.0;
   }


   /* we also need a block to keep track of the diagonal values and a sum */
   diagonal_block =  hypre_CTAlloc(double, bnnz);
   sum_block =  hypre_CTAlloc(double, bnnz);
   distribute_block =  hypre_CTAlloc(double, bnnz);

   /*-----------------------------------------------------------------------
    *  Send and receive fine_to_coarse info.
    *-----------------------------------------------------------------------*/ 

   if (debug_flag==4) wall_time = time_getWallclockSeconds();

   fine_to_coarse_offd = hypre_CTAlloc(int, num_cols_A_offd); 

#define HYPRE_SMP_PRIVATE i,j,ns,ne,size,rest,coarse_shift
#include "../utilities/hypre_smp_forloop.h"
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

   /* again, we do not need to use the block version of comm handle since
      the fine to coarse mapping is size of the nodes */
	
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

#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
   for (i = 0; i < n_fine; i++) fine_to_coarse[i] -= my_first_cpt;

   /*-----------------------------------------------------------------------
    *  Loop over fine grid points.
    *-----------------------------------------------------------------------*/
    
#define HYPRE_SMP_PRIVATE i,j,jl,i1,i2,jj,jj1,ns,ne,size,rest,sum_block,diagonal_block,distribute_block,P_marker,P_marker_offd,strong_f_marker,jj_counter,jj_counter_offd,c_num,jj_begin_row,jj_end_row,jj_begin_row_offd,jj_end_row_offd
#include "../utilities/hypre_smp_forloop.h"
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

     P_marker = hypre_CTAlloc(int, n_fine);
     P_marker_offd = hypre_CTAlloc(int, num_cols_A_offd);

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
         /* P_diag_data[jj_counter] = one; */
         hypre_CSRBlockMatrixBlockCopyData(identity_block,  
                                           &P_diag_data[jj_counter*bnnz], 
                                           1.0, block_size);
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
               /* P_diag_data[jj_counter] = zero; */
               hypre_CSRBlockMatrixBlockCopyData(zero_block,  
                                                 &P_diag_data[jj_counter*bnnz], 
                                                 1.0, block_size);
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
                  P_offd_j[jj_counter_offd]  = i1;
                  /* P_offd_data[jj_counter_offd] = zero; */
                  hypre_CSRBlockMatrixBlockCopyData(zero_block,  
                                                    &P_offd_data[jj_counter_offd*bnnz], 
                                                    1.0, block_size);
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
                  P_offd_j[jj_counter_offd]  = i1;
                  /* P_offd_data[jj_counter_offd] = zero; */
                  hypre_CSRBlockMatrixBlockCopyData(zero_block,  
                                                    &P_offd_data[jj_counter_offd*bnnz], 
                                                    1.0, block_size);

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
         

         /* get the diagonal block */
         /* diagonal = A_diag_data[A_diag_i[i]]; */
         hypre_CSRBlockMatrixBlockCopyData(&A_diag_data[A_diag_i[i]*bnnz], diagonal_block, 
                                           1.0, block_size);
         


          /* Here we go through the neighborhood of this grid point */
     
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
               /*   P_diag_data[P_marker[i1]] += A_diag_data[jj]; */
               hypre_CSRBlockMatrixBlockAddAccumulate(&A_diag_data[jj*bnnz], 
                                                      &P_diag_data[P_marker[i1]*bnnz], 
                                                      block_size);
               
            }

            /*--------------------------------------------------------------
             * Case 2: neighbor i1 is an F-point and strongly influences i,
             * distribute a_{i,i1} to C-points that strongly infuence i.
             * Note: currently no distribution to the diagonal in this case.
             *--------------------------------------------------------------*/
            
            else if (P_marker[i1] == strong_f_marker)
            {
               /* initialize sum to zero */
               /* sum = zero; */
               hypre_CSRBlockMatrixBlockCopyData(zero_block, sum_block, 1.0, 
                                                 block_size);
               
               
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
                     /* add diag data to sum */ 
                     /* sum += A_diag_data[jj1]; */
                     hypre_CSRBlockMatrixBlockAddAccumulate(&A_diag_data[jj1*bnnz], 
                                                            sum_block, block_size);
                  }
               }

               /* Off-Diagonal block part of row i1 */ 
               if (num_procs > 1)
               {              
                  for (jj1 = A_offd_i[i1]; jj1 < A_offd_i[i1+1]; jj1++)
                  {
                     i2 = A_offd_j[jj1];
                     if (P_marker_offd[i2] >= jj_begin_row_offd )
                     {
                        /* add off diag data to sum */ 
                        /*sum += A_offd_data[jj1];*/
                        hypre_CSRBlockMatrixBlockAddAccumulate(&A_offd_data[jj1*bnnz], 
                                                               sum_block, block_size);

                     }
                  }
               } 
               /* check whether sum_block is singular */

                  /* distribute = A_diag_data[jj] / sum;*/
                  /* here we want: A_diag_data * sum^(-1) */
                  /* note that results are uneffected for most problems if
                     we do sum^(-1) * A_diag_data - but it seems to matter
                     a little for very non-sym */  

               if (hypre_CSRBlockMatrixBlockMultInv(sum_block, &A_diag_data[jj*bnnz], 
                                                    distribute_block, block_size) == 0)
               {
                     
 
                  /*-----------------------------------------------------------
                   * Loop over row of A for point i1 and do the distribution.
                   *-----------------------------------------------------------*/
                  
                  /* Diagonal block part of row i1 */
                  for (jj1 = A_diag_i[i1]; jj1 < A_diag_i[i1+1]; jj1++)
                  {
                     i2 = A_diag_j[jj1];
                     if (P_marker[i2] >= jj_begin_row )
                     {
                       
                        /*  P_diag_data[P_marker[i2]]
                            += distribute * A_diag_data[jj1];*/

                        /* multiply - result in sum_block */ 
                        hypre_CSRBlockMatrixBlockMultAdd(distribute_block, 
                                                         &A_diag_data[jj1*bnnz], 0.0, 
                                                         sum_block, block_size);
                        

                        /* add result to p_diag_data */
                        hypre_CSRBlockMatrixBlockAddAccumulate(sum_block, 
                                                               &P_diag_data[P_marker[i2]*bnnz], 
                                                               block_size);

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
                           /* P_offd_data[P_marker_offd[i2]]    
                              += distribute * A_offd_data[jj1]; */ 

                           /* multiply - result in sum_block */ 
                           hypre_CSRBlockMatrixBlockMultAdd(distribute_block, 
                                                            &A_offd_data[jj1*bnnz], 0.0, 
                                                            sum_block, block_size);
                           
                           
                           /* add result to p_offd_data */
                           hypre_CSRBlockMatrixBlockAddAccumulate(sum_block, 
                                                                  &P_offd_data[P_marker_offd[i2]*bnnz], 
                                                                  block_size);
                        }
                     }
                  }
               }
               else /* sum block is all zeros (or almost singular) - just add to diagonal */
               {
                  /* diagonal += A_diag_data[jj]; */
                  hypre_CSRBlockMatrixBlockAddAccumulate(&A_diag_data[jj*bnnz], 
                                                         diagonal_block, 
                                                         block_size);

               }
            }
            
            /*--------------------------------------------------------------
             * Case 3: neighbor i1 weakly influences i, accumulate a_{i,i1}
             * into the diagonal.
             *--------------------------------------------------------------*/

            else if (CF_marker[i1] != -3)
            {
               /* diagonal += A_diag_data[jj];*/
               hypre_CSRBlockMatrixBlockAddAccumulate(&A_diag_data[jj*bnnz], 
                                                      diagonal_block, 
                                                      block_size);
               
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
                  /* P_offd_data[P_marker_offd[i1]] += A_offd_data[jj]; */
                  hypre_CSRBlockMatrixBlockAddAccumulate( &A_offd_data[jj*bnnz],
                                                          &P_offd_data[P_marker_offd[i1]*bnnz],
                                                          block_size);
               }

               /*------------------------------------------------------------
                * Case 2: neighbor i1 is an F-point and strongly influences i,
                * distribute a_{i,i1} to C-points that strongly infuence i.
                * Note: currently no distribution to the diagonal in this case.
                *-----------------------------------------------------------*/
            
               else if (P_marker_offd[i1] == strong_f_marker)
               {
                  
                  /* initialize sum to zero */
                  hypre_CSRBlockMatrixBlockCopyData(zero_block, sum_block, 
                                                    1.0, block_size);
                  
                  /*---------------------------------------------------------
                   * Loop over row of A_ext for point i1 and calculate the sum
                   * of the connections to c-points that strongly influence i.
                   *---------------------------------------------------------*/
                  
                  /* find row number */
                  c_num = A_offd_j[jj];
                  
                  for (jj1 = A_ext_i[c_num]; jj1 < A_ext_i[c_num+1]; jj1++)
                  {
                     i2 = A_ext_j[jj1];
                     
                     if (i2 > -1)
                     {                            
                                           /* in the diagonal block */
                        if (P_marker[i2] >= jj_begin_row)
                        {
                           /* sum += A_ext_data[jj1]; */
                           hypre_CSRBlockMatrixBlockAddAccumulate(&A_ext_data[jj1*bnnz], 
                                         sum_block, block_size);
                        }
                     }
                     else                       
                     {                          
                                           /* in the off_diagonal block  */
                        if (P_marker_offd[-i2-1] >= jj_begin_row_offd)
                        {
                           /* sum += A_ext_data[jj1]; */
                           hypre_CSRBlockMatrixBlockAddAccumulate(&A_ext_data[jj1*bnnz], 
                                                                  sum_block, block_size);
                           
                        }
                     }
                  }

                  /* check whether sum_block is singular */
                  
                  
                  /* distribute = A_offd_data[jj] / sum;  */
                  /* here we want: A_offd_data * sum^(-1) */
                  if (hypre_CSRBlockMatrixBlockMultInv(sum_block, &A_offd_data[jj*bnnz], 
                                                       distribute_block, block_size) == 0)
                  {
                     
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
                           if (P_marker[i2] >= jj_begin_row)
                           {
                              /* P_diag_data[P_marker[i2]]
                                 += distribute * A_ext_data[jj1]; */

                              /* multiply - result in sum_block */ 
                              hypre_CSRBlockMatrixBlockMultAdd(distribute_block, 
                                                               &A_ext_data[jj1*bnnz], 0.0, 
                                                               sum_block, block_size);
                              

                              /* add result to p_diag_data */
                              hypre_CSRBlockMatrixBlockAddAccumulate(sum_block, 
                                                                     &P_diag_data[P_marker[i2]*bnnz], 
                                                                     block_size);

                           }
                        }
                        else
                        {
                           /* in the off_diagonal block  */
                           if (P_marker_offd[-i2-1] >= jj_begin_row_offd)

                              /*P_offd_data[P_marker_offd[-i2-1]]
                                += distribute * A_ext_data[jj1];*/
                           {
                              
                              /* multiply - result in sum_block */ 
                              hypre_CSRBlockMatrixBlockMultAdd(distribute_block, 
                                                               &A_ext_data[jj1*bnnz], 0.0, 
                                                               sum_block, block_size);
                           
                           
                              /* add result to p_offd_data */
                              hypre_CSRBlockMatrixBlockAddAccumulate(sum_block, 
                                                                     &P_offd_data[P_marker_offd[-i2-1]*bnnz], 
                                                                     block_size);
                           }
                           

                        }
                     }
                  }
		  else /* sum block is all zeros - just add to diagonal */
                  {
                     /* diagonal += A_offd_data[jj]; */
                     hypre_CSRBlockMatrixBlockAddAccumulate(&A_offd_data[jj*bnnz], 
                                                            diagonal_block, 
                                                            block_size);

                  }
               }
            
               /*-----------------------------------------------------------
                * Case 3: neighbor i1 weakly influences i, accumulate a_{i,i1}
                * into the diagonal.
                *-----------------------------------------------------------*/

               else if (CF_marker_offd[i1] != -3)
               {
                  /* diagonal += A_offd_data[jj]; */
                  hypre_CSRBlockMatrixBlockAddAccumulate(&A_offd_data[jj*bnnz], 
                                                         diagonal_block, 
                                                         block_size);

               } 
            }
         }           

        /*-----------------------------------------------------------------
          * Set interpolation weight by dividing by the diagonal.
          *-----------------------------------------------------------------*/

         for (jj = jj_begin_row; jj < jj_end_row; jj++)
         {

            /* P_diag_data[jj] /= -diagonal; */
              
            /* want diagonal^(-1)*P_diag_data */
            /* do division - put in sum_block */
            if ( hypre_CSRBlockMatrixBlockInvMult(diagonal_block, &P_diag_data[jj*bnnz], 
                                                  sum_block, block_size) == 0)
            {
               /* now copy to  P_diag_data[jj] and make negative */
               hypre_CSRBlockMatrixBlockCopyData(sum_block, &P_diag_data[jj*bnnz], 
                                                 -1.0, block_size);
            }
            else
            {
               /* printf(" Warning! singular diagonal block! Proc id %d row %d\n", my_id,i);  */
               /* just make P_diag_data negative since diagonal is singular) */   
               hypre_CSRBlockMatrixBlockCopyData(&P_diag_data[jj*bnnz], &P_diag_data[jj*bnnz], 
                                                 -1.0, block_size);

            }
         }

         for (jj = jj_begin_row_offd; jj < jj_end_row_offd; jj++)
         {
            /* P_offd_data[jj] /= -diagonal; */

            /* do division - put in sum_block */
            hypre_CSRBlockMatrixBlockInvMult(diagonal_block, &P_offd_data[jj*bnnz], 
                                             sum_block, block_size);
            
            /* now copy to  P_offd_data[jj] and make negative */
            hypre_CSRBlockMatrixBlockCopyData(sum_block, &P_offd_data[jj*bnnz], 
                                              -1.0, block_size);
            


         }
           
      }

      strong_f_marker--; 

      P_offd_i[i+1] = jj_counter_offd;
     }
     hypre_TFree(P_marker);
     hypre_TFree(P_marker_offd);
   }

   /* copy row starts since A will be destroyed */
   A_col_starts = hypre_ParCSRBlockMatrixColStarts(A);

#ifdef HYPRE_NO_GLOBAL_PARTITION
   P_row_starts = hypre_CTAlloc(int, 2); /* don't free this */
   for (i = 0; i < 2; i++)
   {
      P_row_starts[i] =  A_col_starts[i];
   }
#else
   P_row_starts = hypre_CTAlloc(int, num_procs + 1); /* don't free this */
   for (i = 0; i < num_procs + 1; i++)
   {
      P_row_starts[i] =  A_col_starts[i];
   }
#endif

   /* Now create P - as a block matrix */
   P = hypre_ParCSRBlockMatrixCreate(comm, block_size,
                                hypre_ParCSRBlockMatrixGlobalNumRows(A),
                                total_global_cpts,
                                P_row_starts,
                                num_cpts_global,
                                0,
                                P_diag_i[n_fine],
                                P_offd_i[n_fine]);


   P_diag = hypre_ParCSRBlockMatrixDiag(P);
   hypre_CSRBlockMatrixData(P_diag) = P_diag_data;
   hypre_CSRBlockMatrixI(P_diag) = P_diag_i;
   hypre_CSRBlockMatrixJ(P_diag) = P_diag_j;
 
   P_offd = hypre_ParCSRBlockMatrixOffd(P);
   hypre_CSRBlockMatrixData(P_offd) = P_offd_data;
   hypre_CSRBlockMatrixI(P_offd) = P_offd_i;
   hypre_CSRBlockMatrixJ(P_offd) = P_offd_j;




   /* Compress P, removing coefficients smaller than trunc_factor * Max */
   if (trunc_factor != 0.0)
   {
      hypre_BoomerAMGBlockInterpTruncation(P, trunc_factor);
      P_diag_data = hypre_CSRBlockMatrixData(P_diag);
      P_diag_i = hypre_CSRBlockMatrixI(P_diag);
      P_diag_j = hypre_CSRBlockMatrixJ(P_diag);
      P_offd_data = hypre_CSRBlockMatrixData(P_offd);
      P_offd_i = hypre_CSRBlockMatrixI(P_offd);
      P_offd_j = hypre_CSRBlockMatrixJ(P_offd);
      P_diag_size = P_diag_i[n_fine];
      P_offd_size = P_offd_i[n_fine];
   }


   num_cols_P_offd = 0;
   if (P_offd_size)
   {
      P_marker = hypre_CTAlloc(int, num_cols_A_offd);

#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
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

      col_map_offd_P = hypre_CTAlloc(int,num_cols_P_offd);

      index = 0;
      for (i=0; i < num_cols_P_offd; i++)
      {
         while (P_marker[index]==0) index++;
         col_map_offd_P[i] = index++;
      }

#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
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
   	hypre_ParCSRBlockMatrixColMapOffd(P) = col_map_offd_P;
   	hypre_CSRBlockMatrixNumCols(P_offd) = num_cols_P_offd;
   } 

   /* use block version */
   hypre_GetCommPkgBlockRTFromCommPkgBlockA(P,A, fine_to_coarse_offd);


   *P_ptr = P;


   hypre_TFree(zero_block);
   hypre_TFree(identity_block);
   hypre_TFree(diagonal_block);
   hypre_TFree(sum_block);
   hypre_TFree(distribute_block);

   hypre_TFree(CF_marker_offd);
   hypre_TFree(int_buf_data);
   hypre_TFree(fine_to_coarse);
   hypre_TFree(fine_to_coarse_offd);
   hypre_TFree(coarse_counter);
   hypre_TFree(jj_count);
   hypre_TFree(jj_count_offd);

   if (num_procs > 1) hypre_CSRBlockMatrixDestroy(A_ext);

   return(0);  

}            
    

int
hypre_BoomerAMGBlockInterpTruncation( hypre_ParCSRBlockMatrix *P,
				 double trunc_factor)        
{
   hypre_CSRBlockMatrix *P_diag = hypre_ParCSRBlockMatrixDiag(P);
   int *P_diag_i = hypre_CSRBlockMatrixI(P_diag);
   int *P_diag_j = hypre_CSRBlockMatrixJ(P_diag);
   double *P_diag_data = hypre_CSRBlockMatrixData(P_diag);
   int *P_diag_j_new;
   double *P_diag_data_new;

   hypre_CSRBlockMatrix *P_offd = hypre_ParCSRBlockMatrixOffd(P);
   int *P_offd_i = hypre_CSRBlockMatrixI(P_offd);
   int *P_offd_j = hypre_CSRBlockMatrixJ(P_offd);
   double *P_offd_data = hypre_CSRBlockMatrixData(P_offd);
   int *P_offd_j_new;
   double *P_offd_data_new;

   int  block_size = hypre_CSRBlockMatrixBlockSize(P_diag);
   int  bnnz = block_size*block_size;

   int n_fine = hypre_CSRBlockMatrixNumRows(P_diag);
   int i, j, start_j, k;
   int ierr = 0;
   int next_open = 0;
   int now_checking = 0;
   int num_lost = 0;
   int next_open_offd = 0;
   int now_checking_offd = 0;
   int num_lost_offd = 0;
   int P_diag_size;
   int P_offd_size;
   double max_coef, tmp;
   double *row_sum;
   double *scale;
   double *out_block;
   

   /* for now we will use the frobenius norm to 
      determine whether to keep a block or not  - so norm_type = 1*/
   row_sum  = hypre_CTAlloc(double, bnnz);
   scale = hypre_CTAlloc(double, bnnz);
   out_block = hypre_CTAlloc(double, bnnz);

   /* go through each row */
   for (i = 0; i < n_fine; i++)
   {
      max_coef = 0.0;

      /* diag */
      for (j = P_diag_i[i]; j < P_diag_i[i+1]; j++)
      {
         hypre_CSRBlockMatrixBlockNorm(1, &P_diag_data[j*bnnz] , &tmp, block_size);
         max_coef = (max_coef < tmp) ?  tmp : max_coef;
      }
      
      /* off_diag */ 
      for (j = P_offd_i[i]; j < P_offd_i[i+1]; j++)
      {
         hypre_CSRBlockMatrixBlockNorm(1, &P_offd_data[j*bnnz], &tmp, block_size);
         max_coef = (max_coef < tmp) ?  tmp : max_coef;
      }
      
      max_coef *= trunc_factor;

      start_j = P_diag_i[i];
      P_diag_i[i] -= num_lost;
    
      /* set scale and row sum to zero */
      hypre_CSRBlockMatrixBlockSetScalar(scale, 0.0, block_size);
      hypre_CSRBlockMatrixBlockSetScalar(row_sum, 0.0, block_size);

      for (j = start_j; j < P_diag_i[i+1]; j++)
      {
         /* row_sum += P_diag_data[now_checking];*/
         hypre_CSRBlockMatrixBlockAddAccumulate(&P_diag_data[now_checking*bnnz], row_sum, block_size);

         hypre_CSRBlockMatrixBlockNorm(1, &P_diag_data[now_checking*bnnz] , &tmp, block_size);

         if ( tmp < max_coef)
         {
            num_lost++;
            now_checking++;
         }
         else
         {
	    /* scale += P_diag_data[now_checking]; */
            hypre_CSRBlockMatrixBlockAddAccumulate(&P_diag_data[now_checking*bnnz], scale, block_size);

            /* P_diag_data[next_open] = P_diag_data[now_checking]; */
            hypre_CSRBlockMatrixBlockCopyData( &P_diag_data[now_checking*bnnz], &P_diag_data[next_open*bnnz], 
                                               1.0, block_size);
            
            P_diag_j[next_open] = P_diag_j[now_checking];
            now_checking++;
            next_open++;
         }
      }

      start_j = P_offd_i[i];
      P_offd_i[i] -= num_lost_offd;

      for (j = start_j; j < P_offd_i[i+1]; j++)
      {
	 /* row_sum += P_offd_data[now_checking_offd]; */
         hypre_CSRBlockMatrixBlockAddAccumulate(&P_offd_data[now_checking_offd*bnnz], row_sum, block_size);

         hypre_CSRBlockMatrixBlockNorm(1, &P_offd_data[now_checking_offd*bnnz] , &tmp, block_size);

         if ( tmp < max_coef)
         {
            num_lost_offd++;
            now_checking_offd++;
         }
         else
         {
	    /* scale += P_offd_data[now_checking_offd]; */
            hypre_CSRBlockMatrixBlockAddAccumulate(&P_offd_data[now_checking_offd*bnnz], scale, block_size);

            /* P_offd_data[next_open_offd] = P_offd_data[now_checking_offd];*/
            hypre_CSRBlockMatrixBlockCopyData( &P_offd_data[now_checking_offd*bnnz], &P_offd_data[next_open_offd*bnnz], 
                                               1.0, block_size);
            

            P_offd_j[next_open_offd] = P_offd_j[now_checking_offd];
            now_checking_offd++;
            next_open_offd++;
         }
      }
      /* normalize row of P */

      /* out_block = row_sum/scale; */
      if (hypre_CSRBlockMatrixBlockInvMult(scale, row_sum, out_block, block_size) == 0)
      {
         	 
         for (j = P_diag_i[i]; j < (P_diag_i[i+1]-num_lost); j++)
         {
            /* P_diag_data[j] *= out_block; */

            /* put mult result in row_sum */ 
           hypre_CSRBlockMatrixBlockMultAdd(out_block, &P_diag_data[j*bnnz], 0.0,
                                            row_sum, block_size);
           /* add to P_diag_data */
           hypre_CSRBlockMatrixBlockAddAccumulate(row_sum, &P_diag_data[j*bnnz], block_size);
         }
         
         for (j = P_offd_i[i]; j < (P_offd_i[i+1]-num_lost_offd); j++)
         {
            
            /* P_offd_data[j] *= out_block; */

            /* put mult result in row_sum */ 
            hypre_CSRBlockMatrixBlockMultAdd(out_block, &P_offd_data[j*bnnz], 0.0,
                                             row_sum, block_size);
           /* add to to P_offd_data */
           hypre_CSRBlockMatrixBlockAddAccumulate(row_sum, &P_offd_data[j*bnnz], block_size);

         }
         
      }
   }

   hypre_TFree(row_sum);
   hypre_TFree(scale);
   hypre_TFree(out_block);

   P_diag_i[n_fine] -= num_lost;
   P_offd_i[n_fine] -= num_lost_offd;

   if (num_lost)
   {
      P_diag_size = P_diag_i[n_fine];
      P_diag_j_new = hypre_CTAlloc(int, P_diag_size);
      P_diag_data_new = hypre_CTAlloc(double, P_diag_size*bnnz);
      for (i=0; i < P_diag_size; i++)
      {
	 P_diag_j_new[i] = P_diag_j[i];
         for (k=0; k < bnnz; k++)
         {
            P_diag_data_new[i*bnnz+k] = P_diag_data[i*bnnz+k];
         }
         
      }
      hypre_TFree(P_diag_j);
      hypre_TFree(P_diag_data);
      hypre_CSRMatrixJ(P_diag) = P_diag_j_new;
      hypre_CSRMatrixData(P_diag) = P_diag_data_new;
      hypre_CSRMatrixNumNonzeros(P_diag) = P_diag_size;
   }
   if (num_lost_offd)
   {
      P_offd_size = P_offd_i[n_fine];
      P_offd_j_new = hypre_CTAlloc(int, P_offd_size);
      P_offd_data_new = hypre_CTAlloc(double, P_offd_size*bnnz);
      for (i=0; i < P_offd_size; i++)
      {
         P_offd_j_new[i] = P_offd_j[i];
         for (k=0; k < bnnz; k++)
         {
            P_offd_data_new[i*bnnz + k] = P_offd_data[i*bnnz + k];
         }
         
      }
      hypre_TFree(P_offd_j);
      hypre_TFree(P_offd_data);
      hypre_CSRMatrixJ(P_offd) = P_offd_j_new;
      hypre_CSRMatrixData(P_offd) = P_offd_data_new;
      hypre_CSRMatrixNumNonzeros(P_offd) = P_offd_size;
   }
   return ierr;
}


/*---------------------------------------------------------------------------
 * hypre_BoomerAMGBlockBuildInterpDiag

   This is the block version of classical R-S interpolation. We use just the 
   diagonals of these blocks.

   A and P are now Block matrices.  The Strength matrix S is not as it gives
   nodal strengths.

   CF_marker is size number of nodes.

 *--------------------------------------------------------------------------*/

int
hypre_BoomerAMGBuildBlockInterpDiag( hypre_ParCSRBlockMatrix   *A,
                         int                  *CF_marker,
                         hypre_ParCSRMatrix   *S,
                         int                  *num_cpts_global,
                         int                   num_functions,
                         int                  *dof_func,
                         int                   debug_flag,
                         double                trunc_factor,
                         int 		      *col_offd_S_to_A,
                         hypre_ParCSRBlockMatrix  **P_ptr)
{

   MPI_Comm 	      comm = hypre_ParCSRBlockMatrixComm(A);   
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRBlockMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;

   hypre_CSRBlockMatrix *A_diag = hypre_ParCSRBlockMatrixDiag(A);
   double               *A_diag_data = hypre_CSRBlockMatrixData(A_diag);
   int                  *A_diag_i = hypre_CSRBlockMatrixI(A_diag);
   int                  *A_diag_j = hypre_CSRBlockMatrixJ(A_diag);

   int                  block_size = hypre_CSRBlockMatrixBlockSize(A_diag);
   int                  bnnz = block_size*block_size;
   
   hypre_CSRBlockMatrix *A_offd = hypre_ParCSRBlockMatrixOffd(A);   
   double          *A_offd_data = hypre_CSRBlockMatrixData(A_offd);
   int             *A_offd_i = hypre_CSRBlockMatrixI(A_offd);
   int             *A_offd_j = hypre_CSRBlockMatrixJ(A_offd);
   int              num_cols_A_offd = hypre_CSRBlockMatrixNumCols(A_offd);
   int             *col_map_offd = hypre_ParCSRBlockMatrixColMapOffd(A);

   hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
   int             *S_diag_i = hypre_CSRMatrixI(S_diag);
   int             *S_diag_j = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix *S_offd = hypre_ParCSRMatrixOffd(S);   
   int             *S_offd_i = hypre_CSRMatrixI(S_offd);
   int             *S_offd_j = hypre_CSRMatrixJ(S_offd);

   hypre_ParCSRBlockMatrix *P;
   int		           *col_map_offd_P;

   int             *CF_marker_offd;

   hypre_CSRBlockMatrix *A_ext;
   
   double          *A_ext_data;
   int             *A_ext_i;
   int             *A_ext_j;

   hypre_CSRBlockMatrix    *P_diag;
   hypre_CSRBlockMatrix    *P_offd;   

   double          *P_diag_data;
   int             *P_diag_i;
   int             *P_diag_j;
   double          *P_offd_data;
   int             *P_offd_i;
   int             *P_offd_j;

   int              P_diag_size, P_offd_size;
   
   int             *P_marker, *P_marker_offd;

   int              jj_counter,jj_counter_offd;
   int             *jj_count, *jj_count_offd;
   int              jj_begin_row,jj_begin_row_offd;
   int              jj_end_row,jj_end_row_offd;
   
   int              start_indexing = 0; /* start indexing for P_data at 0 */

   int              n_fine = hypre_CSRBlockMatrixNumRows(A_diag);

   int              strong_f_marker;

   int             *fine_to_coarse;
   int             *fine_to_coarse_offd;
   int             *coarse_counter;
   int              coarse_shift;
   int              total_global_cpts;
   int              num_cols_P_offd, my_first_cpt;

   int              bd;
   
   int              i,i1,i2;
   int              j,jl,jj,jj1;
   int              k,kc;
   int              start;

   int              c_num;
   
   int              my_id;
   int              num_procs;
   int              num_threads;
   int              num_sends;
   int              index;
   int              ns, ne, size, rest;
   int             *int_buf_data;

   int col_1 = hypre_ParCSRBlockMatrixFirstRowIndex(A);
   int local_numrows = hypre_CSRBlockMatrixNumRows(A_diag);
   int col_n = col_1 + local_numrows;

   double           wall_time;  /* for debugging instrumentation  */

   
   double           *identity_block;
   double           *zero_block;
   double           *diagonal_block;
   double           *sum_block;
   double           *distribute_block;
   
   int               *P_row_starts, *A_col_starts;
   

   MPI_Comm_size(comm, &num_procs);   
   MPI_Comm_rank(comm,&my_id);
   num_threads = hypre_NumThreads();


#ifdef HYPRE_NO_GLOBAL_PARTITION
   my_first_cpt = num_cpts_global[0];
   if (my_id == (num_procs -1)) total_global_cpts = num_cpts_global[1];
   MPI_Bcast(&total_global_cpts, 1, MPI_INT, num_procs-1, comm);
#else
   my_first_cpt = num_cpts_global[my_id];
   total_global_cpts = num_cpts_global[num_procs];
#endif

   /*-------------------------------------------------------------------
    * Get the CF_marker data for the off-processor columns
    *-------------------------------------------------------------------*/

   if (debug_flag==4) wall_time = time_getWallclockSeconds();

   CF_marker_offd = hypre_CTAlloc(int, num_cols_A_offd);


   if (!comm_pkg)
   {
#ifdef HYPRE_NO_GLOBAL_PARTITION
        hypre_BlockNewCommPkgCreate(A); 
#else
	hypre_BlockMatvecCommPkgCreate(A);
#endif
	comm_pkg = hypre_ParCSRBlockMatrixCommPkg(A); 
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   int_buf_data = hypre_CTAlloc(int, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
						num_sends));

   index = 0;
   for (i = 0; i < num_sends; i++)
   {
	start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
	for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
        {
           int_buf_data[index++] 
                   = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
        }
        
   }
	
   /* we do not need the block version of comm handle - because
      CF_marker corresponds to the nodal matrix.  This call populates
      CF_marker_offd */
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
      A_ext      = hypre_ParCSRBlockMatrixExtractBExt(A, A, 1);
      A_ext_i    = hypre_CSRBlockMatrixI(A_ext);
      A_ext_j    = hypre_CSRBlockMatrixJ(A_ext);
      A_ext_data = hypre_CSRBlockMatrixData(A_ext);
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
            /* for the data field we must get all of the block data */    
            for (bd = 0; bd < bnnz; bd++)
            {
               A_ext_data[index*bnnz + bd] = A_ext_data[j*bnnz + bd];
            }
            index++;
         }
         else
         {
            kc = hypre_BinarySearch(col_map_offd, k, num_cols_A_offd);
            if (kc > -1)
            {
               A_ext_j[index] = -kc-1;
               for (bd = 0; bd < bnnz; bd++)
               {
                  A_ext_data[index*bnnz + bd] = A_ext_data[j*bnnz + bd];
               }
               index++;
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
      printf("Proc = %d  Interp: Comm 2   Get A_ext =  %f\n",
                    my_id, wall_time);
      fflush(NULL);
   }


   /*-----------------------------------------------------------------------
    *  First Pass: Determine size of P and fill in fine_to_coarse mapping.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Intialize counters and allocate mapping vector.
    *-----------------------------------------------------------------------*/

   coarse_counter = hypre_CTAlloc(int, num_threads);
   jj_count = hypre_CTAlloc(int, num_threads);
   jj_count_offd = hypre_CTAlloc(int, num_threads);

   fine_to_coarse = hypre_CTAlloc(int, n_fine);
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
   for (i = 0; i < n_fine; i++) fine_to_coarse[i] = -1;

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;
      
   /*-----------------------------------------------------------------------
    *  Loop over fine grid.
    *-----------------------------------------------------------------------*/

/* RDF: this looks a little tricky, but doable */
#define HYPRE_SMP_PRIVATE i,j,i1,jj,ns,ne,size,rest
#include "../utilities/hypre_smp_forloop.h"
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


     /* loop over the fine grid points */   
     for (i = ns; i < ne; i++)
     {
      
      /*--------------------------------------------------------------------
       *  If i is a C-point, interpolation is the identity. Also set up
       *  mapping vector (fine_to_coarse is the mapping vector).
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

   P_diag_i    = hypre_CTAlloc(int, n_fine+1);
   P_diag_j    = hypre_CTAlloc(int, P_diag_size);
   /* we need to include the size of the blocks in the data size */
   P_diag_data = hypre_CTAlloc(double, P_diag_size*bnnz);

   P_diag_i[n_fine] = jj_counter; 


   P_offd_size = jj_counter_offd;

   P_offd_i    = hypre_CTAlloc(int, n_fine+1);
   P_offd_j    = hypre_CTAlloc(int, P_offd_size);
   /* we need to include the size of the blocks in the data size */
   P_offd_data = hypre_CTAlloc(double, P_offd_size*bnnz);

   /*-----------------------------------------------------------------------
    *  Intialize some stuff.
    *-----------------------------------------------------------------------*/

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   if (debug_flag==4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      printf("Proc = %d     Interp: Internal work 1 =     %f\n",
                    my_id, wall_time);
      fflush(NULL);
   }

   /* we need a block identity and a block of zeros*/
   identity_block = hypre_CTAlloc(double, bnnz);
   zero_block =  hypre_CTAlloc(double, bnnz);
 
   for(i = 0; i < block_size; i++) 
   {
      identity_block[i*block_size + i] = 1.0;
   }


   /* we also need a block to keep track of the diagonal values and a sum */
   diagonal_block =  hypre_CTAlloc(double, bnnz);
   sum_block =  hypre_CTAlloc(double, bnnz);
   distribute_block =  hypre_CTAlloc(double, bnnz);

   /*-----------------------------------------------------------------------
    *  Send and receive fine_to_coarse info.
    *-----------------------------------------------------------------------*/ 

   if (debug_flag==4) wall_time = time_getWallclockSeconds();

   fine_to_coarse_offd = hypre_CTAlloc(int, num_cols_A_offd); 

#define HYPRE_SMP_PRIVATE i,j,ns,ne,size,rest,coarse_shift
#include "../utilities/hypre_smp_forloop.h"
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

   /* again, we do not need to use the block version of comm handle since
      the fine to coarse mapping is size of the nodes */
	
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

#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
   for (i = 0; i < n_fine; i++) fine_to_coarse[i] -= my_first_cpt;

   /*-----------------------------------------------------------------------
    *  Loop over fine grid points.
    *-----------------------------------------------------------------------*/
    
#define HYPRE_SMP_PRIVATE i,j,jl,i1,i2,jj,jj1,ns,ne,size,rest,sum_block,diagonal_block,distribute_block,P_marker,P_marker_offd,strong_f_marker,jj_counter,jj_counter_offd,c_num,jj_begin_row,jj_end_row,jj_begin_row_offd,jj_end_row_offd
#include "../utilities/hypre_smp_forloop.h"
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

     P_marker = hypre_CTAlloc(int, n_fine);
     P_marker_offd = hypre_CTAlloc(int, num_cols_A_offd);

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
         /* P_diag_data[jj_counter] = one; */
         hypre_CSRBlockMatrixBlockCopyData(identity_block,  
                                           &P_diag_data[jj_counter*bnnz], 
                                           1.0, block_size);
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
               /* P_diag_data[jj_counter] = zero; */
               hypre_CSRBlockMatrixBlockCopyData(zero_block,  
                                                 &P_diag_data[jj_counter*bnnz], 
                                                 1.0, block_size);
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
                  P_offd_j[jj_counter_offd]  = i1;
                  /* P_offd_data[jj_counter_offd] = zero; */
                  hypre_CSRBlockMatrixBlockCopyData(zero_block,  
                                                    &P_offd_data[jj_counter_offd*bnnz], 
                                                    1.0, block_size);
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
                  P_offd_j[jj_counter_offd]  = i1;
                  /* P_offd_data[jj_counter_offd] = zero; */
                  hypre_CSRBlockMatrixBlockCopyData(zero_block,  
                                                    &P_offd_data[jj_counter_offd*bnnz], 
                                                    1.0, block_size);

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
         

         /* get the diagonal block */
         /* diagonal = A_diag_data[A_diag_i[i]]; */
         hypre_CSRBlockMatrixBlockCopyDataDiag(&A_diag_data[A_diag_i[i]*bnnz], diagonal_block, 
                                           1.0, block_size);
         


          /* Here we go through the neighborhood of this grid point */
     
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
               /*   P_diag_data[P_marker[i1]] += A_diag_data[jj]; */
               hypre_CSRBlockMatrixBlockAddAccumulateDiag(&A_diag_data[jj*bnnz], 
                                                      &P_diag_data[P_marker[i1]*bnnz], 
                                                      block_size);
               
            }

            /*--------------------------------------------------------------
             * Case 2: neighbor i1 is an F-point and strongly influences i,
             * distribute a_{i,i1} to C-points that strongly infuence i.
             * Note: currently no distribution to the diagonal in this case.
             *--------------------------------------------------------------*/
            
            else if (P_marker[i1] == strong_f_marker)
            {
               /* initialize sum to zero */
               /* sum = zero; */
               hypre_CSRBlockMatrixBlockCopyData(zero_block, sum_block, 1.0, 
                                                 block_size);
               
               
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
                     /* add diag data to sum */ 
                     /* sum += A_diag_data[jj1]; */
                     hypre_CSRBlockMatrixBlockAddAccumulateDiag(&A_diag_data[jj1*bnnz], 
                                                            sum_block, block_size);
                  }
               }

               /* Off-Diagonal block part of row i1 */ 
               if (num_procs > 1)
               {              
                  for (jj1 = A_offd_i[i1]; jj1 < A_offd_i[i1+1]; jj1++)
                  {
                     i2 = A_offd_j[jj1];
                     if (P_marker_offd[i2] >= jj_begin_row_offd )
                     {
                        /* add off diag data to sum */ 
                        /*sum += A_offd_data[jj1];*/
                        hypre_CSRBlockMatrixBlockAddAccumulateDiag(&A_offd_data[jj1*bnnz], 
                                                               sum_block, block_size);

                     }
                  }
               } 
               /* check whether sum_block is singular */

                  /* distribute = A_diag_data[jj] / sum;*/
                  /* here we want: A_diag_data * sum^(-1) */

               if (hypre_CSRBlockMatrixBlockInvMultDiag(sum_block, &A_diag_data[jj*bnnz], 
                                                    distribute_block, block_size) == 0)
               {
                     
 
                  /*-----------------------------------------------------------
                   * Loop over row of A for point i1 and do the distribution.
                   *-----------------------------------------------------------*/
                  
                  /* Diagonal block part of row i1 */
                  for (jj1 = A_diag_i[i1]; jj1 < A_diag_i[i1+1]; jj1++)
                  {
                     i2 = A_diag_j[jj1];
                     if (P_marker[i2] >= jj_begin_row )
                     {
                       
                        /*  P_diag_data[P_marker[i2]]
                            += distribute * A_diag_data[jj1];*/

                        /* multiply - result in sum_block */ 
                        hypre_CSRBlockMatrixBlockCopyData(zero_block,  
                                                          sum_block, 1.0, block_size);
                        hypre_CSRBlockMatrixBlockMultAddDiag(distribute_block, 
                                                             &A_diag_data[jj1*bnnz], 0.0, 
                                                             sum_block, block_size);
                        

                        /* add result to p_diag_data */
                        hypre_CSRBlockMatrixBlockAddAccumulateDiag(sum_block, 
                                                               &P_diag_data[P_marker[i2]*bnnz], 
                                                               block_size);

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
                           /* P_offd_data[P_marker_offd[i2]]    
                              += distribute * A_offd_data[jj1]; */ 

                           /* multiply - result in sum_block */ 

                           hypre_CSRBlockMatrixBlockCopyData(zero_block,  
                                                    sum_block, 1.0, block_size);
                           hypre_CSRBlockMatrixBlockMultAddDiag(distribute_block, 
                                                            &A_offd_data[jj1*bnnz], 0.0, 
                                                            sum_block, block_size);
                           
                           
                           /* add result to p_offd_data */
                           hypre_CSRBlockMatrixBlockAddAccumulateDiag(sum_block, 
                                                                  &P_offd_data[P_marker_offd[i2]*bnnz], 
                                                                  block_size);
                        }
                     }
                  }
               }
               else /* sum block is all zeros (or almost singular) - just add to diagonal */
               {
                  /* diagonal += A_diag_data[jj]; */
                  hypre_CSRBlockMatrixBlockAddAccumulateDiag(&A_diag_data[jj*bnnz], 
                                                         diagonal_block, 
                                                         block_size);

               }
            }
            
            /*--------------------------------------------------------------
             * Case 3: neighbor i1 weakly influences i, accumulate a_{i,i1}
             * into the diagonal.
             *--------------------------------------------------------------*/

            else if (CF_marker[i1] != -3)
            {
               /* diagonal += A_diag_data[jj];*/
               hypre_CSRBlockMatrixBlockAddAccumulateDiag(&A_diag_data[jj*bnnz], 
                                                      diagonal_block, 
                                                      block_size);
               
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
                  /* P_offd_data[P_marker_offd[i1]] += A_offd_data[jj]; */
                  hypre_CSRBlockMatrixBlockAddAccumulateDiag( &A_offd_data[jj*bnnz],
                                                              &P_offd_data[P_marker_offd[i1]*bnnz],
                                                              block_size);
               }

               /*------------------------------------------------------------
                * Case 2: neighbor i1 is an F-point and strongly influences i,
                * distribute a_{i,i1} to C-points that strongly infuence i.
                * Note: currently no distribution to the diagonal in this case.
                *-----------------------------------------------------------*/
            
               else if (P_marker_offd[i1] == strong_f_marker)
               {
                  
                  /* initialize sum to zero */
                  hypre_CSRBlockMatrixBlockCopyData(zero_block, sum_block, 
                                                    1.0, block_size);
                  
                  /*---------------------------------------------------------
                   * Loop over row of A_ext for point i1 and calculate the sum
                   * of the connections to c-points that strongly influence i.
                   *---------------------------------------------------------*/
                  
                  /* find row number */
                  c_num = A_offd_j[jj];
                  
                  for (jj1 = A_ext_i[c_num]; jj1 < A_ext_i[c_num+1]; jj1++)
                  {
                     i2 = A_ext_j[jj1];
                     
                     if (i2 > -1)
                     {                            
                                           /* in the diagonal block */
                        if (P_marker[i2] >= jj_begin_row)
                        {
                           /* sum += A_ext_data[jj1]; */
                           hypre_CSRBlockMatrixBlockAddAccumulateDiag(&A_ext_data[jj1*bnnz], 
                                         sum_block, block_size);
                        }
                     }
                     else                       
                     {                          
                                           /* in the off_diagonal block  */
                        if (P_marker_offd[-i2-1] >= jj_begin_row_offd)
                        {
                           /* sum += A_ext_data[jj1]; */
                           hypre_CSRBlockMatrixBlockAddAccumulateDiag(&A_ext_data[jj1*bnnz], 
                                                                  sum_block, block_size);
                           
                        }
                     }
                  }

                  /* check whether sum_block is singular */
                  
                  
                  /* distribute = A_offd_data[jj] / sum;  */
                  /* here we want: A_offd_data * sum^(-1) */
                  if (hypre_CSRBlockMatrixBlockInvMultDiag(sum_block, &A_offd_data[jj*bnnz], 
                                                           distribute_block, block_size) == 0)
                  {
                     
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
                           if (P_marker[i2] >= jj_begin_row)
                           {
                              /* P_diag_data[P_marker[i2]]
                                 += distribute * A_ext_data[jj1]; */

                              /* multiply - result in sum_block */ 
                              hypre_CSRBlockMatrixBlockCopyData(zero_block, sum_block, 1.0, block_size);

                              hypre_CSRBlockMatrixBlockMultAdd(distribute_block, 
                                                               &A_ext_data[jj1*bnnz], 0.0, 
                                                               sum_block, block_size);
                              

                              /* add result to p_diag_data */
                              hypre_CSRBlockMatrixBlockAddAccumulateDiag(sum_block, 
                                                                     &P_diag_data[P_marker[i2]*bnnz], 
                                                                     block_size);

                           }
                        }
                        else
                        {
                           /* in the off_diagonal block  */
                           if (P_marker_offd[-i2-1] >= jj_begin_row_offd)

                              /*P_offd_data[P_marker_offd[-i2-1]]
                                += distribute * A_ext_data[jj1];*/
                           {
                              
                              /* multiply - result in sum_block */ 
                              hypre_CSRBlockMatrixBlockCopyData(zero_block, sum_block, 1.0, block_size);

                              hypre_CSRBlockMatrixBlockMultAddDiag(distribute_block, 
                                                               &A_ext_data[jj1*bnnz], 0.0, 
                                                               sum_block, block_size);
                           
                           
                              /* add result to p_offd_data */
                              hypre_CSRBlockMatrixBlockAddAccumulateDiag(sum_block, 
                                                                         &P_offd_data[P_marker_offd[-i2-1]*bnnz], 
                                                                         block_size);
                           }
                           

                        }
                     }
                  }
		  else /* sum block is all zeros - just add to diagonal */
                  {
                     /* diagonal += A_offd_data[jj]; */
                     hypre_CSRBlockMatrixBlockAddAccumulateDiag(&A_offd_data[jj*bnnz], 
                                                                diagonal_block, 
                                                                block_size);

                  }
               }
            
               /*-----------------------------------------------------------
                * Case 3: neighbor i1 weakly influences i, accumulate a_{i,i1}
                * into the diagonal.
                *-----------------------------------------------------------*/

               else if (CF_marker_offd[i1] != -3)
               {
                  /* diagonal += A_offd_data[jj]; */
                  hypre_CSRBlockMatrixBlockAddAccumulateDiag(&A_offd_data[jj*bnnz], 
                                                         diagonal_block, 
                                                         block_size);

               } 
            }
         }           

        /*-----------------------------------------------------------------
          * Set interpolation weight by dividing by the diagonal.
          *-----------------------------------------------------------------*/

         for (jj = jj_begin_row; jj < jj_end_row; jj++)
         {

            /* P_diag_data[jj] /= -diagonal; */
              
            /* want diagonal^(-1)*P_diag_data */
            /* do division - put in sum_block */
            if ( hypre_CSRBlockMatrixBlockInvMultDiag(diagonal_block, &P_diag_data[jj*bnnz], 
                                                  sum_block, block_size) == 0)
            {
               /* now copy to  P_diag_data[jj] and make negative */
               hypre_CSRBlockMatrixBlockCopyData(sum_block, &P_diag_data[jj*bnnz], 
                                                 -1.0, block_size);
            }
            else
            {
               /* printf(" Warning! singular diagonal block! Proc id %d row %d\n", my_id,i);  */
               /* just make P_diag_data negative since diagonal is zero */   
               hypre_CSRBlockMatrixBlockCopyData(&P_diag_data[jj*bnnz], &P_diag_data[jj*bnnz], 
                                                 -1.0, block_size);

            }
         }

         for (jj = jj_begin_row_offd; jj < jj_end_row_offd; jj++)
         {
            /* P_offd_data[jj] /= -diagonal; */

            /* do division - put in sum_block */
            hypre_CSRBlockMatrixBlockInvMultDiag(diagonal_block, &P_offd_data[jj*bnnz], 
                                             sum_block, block_size);
            
            /* now copy to  P_offd_data[jj] and make negative */
            hypre_CSRBlockMatrixBlockCopyData(sum_block, &P_offd_data[jj*bnnz], 
                                              -1.0, block_size);
            


         }
           
      }

      strong_f_marker--; 

      P_offd_i[i+1] = jj_counter_offd;
     }
     hypre_TFree(P_marker);
     hypre_TFree(P_marker_offd);
   }

   /* copy row starts since A will be destroyed */
   A_col_starts = hypre_ParCSRBlockMatrixColStarts(A);

#ifdef HYPRE_NO_GLOBAL_PARTITION
   P_row_starts = hypre_CTAlloc(int, 2); /* don't free this */
   for (i = 0; i < 2; i++)
   {
      P_row_starts[i] =  A_col_starts[i];
   }
#else
   P_row_starts = hypre_CTAlloc(int, num_procs + 1); /* don't free this */
   for (i = 0; i < num_procs + 1; i++)
   {
      P_row_starts[i] =  A_col_starts[i];
   }
#endif

   /* Now create P - as a block matrix */
   P = hypre_ParCSRBlockMatrixCreate(comm, block_size,
                                hypre_ParCSRBlockMatrixGlobalNumRows(A),
                                total_global_cpts,
                                P_row_starts,
                                num_cpts_global,
                                0,
                                P_diag_i[n_fine],
                                P_offd_i[n_fine]);


   P_diag = hypre_ParCSRBlockMatrixDiag(P);
   hypre_CSRBlockMatrixData(P_diag) = P_diag_data;
   hypre_CSRBlockMatrixI(P_diag) = P_diag_i;
   hypre_CSRBlockMatrixJ(P_diag) = P_diag_j;
 
   P_offd = hypre_ParCSRBlockMatrixOffd(P);
   hypre_CSRBlockMatrixData(P_offd) = P_offd_data;
   hypre_CSRBlockMatrixI(P_offd) = P_offd_i;
   hypre_CSRBlockMatrixJ(P_offd) = P_offd_j;




   /* Compress P, removing coefficients smaller than trunc_factor * Max */
   if (trunc_factor != 0.0)
   {
      hypre_BoomerAMGBlockInterpTruncation(P, trunc_factor);
      P_diag_data = hypre_CSRBlockMatrixData(P_diag);
      P_diag_i = hypre_CSRBlockMatrixI(P_diag);
      P_diag_j = hypre_CSRBlockMatrixJ(P_diag);
      P_offd_data = hypre_CSRBlockMatrixData(P_offd);
      P_offd_i = hypre_CSRBlockMatrixI(P_offd);
      P_offd_j = hypre_CSRBlockMatrixJ(P_offd);
      P_diag_size = P_diag_i[n_fine];
      P_offd_size = P_offd_i[n_fine];
   }


   num_cols_P_offd = 0;
   if (P_offd_size)
   {
      P_marker = hypre_CTAlloc(int, num_cols_A_offd);

#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
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

      col_map_offd_P = hypre_CTAlloc(int,num_cols_P_offd);

      index = 0;
      for (i=0; i < num_cols_P_offd; i++)
      {
         while (P_marker[index]==0) index++;
         col_map_offd_P[i] = index++;
      }

#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
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
   	hypre_ParCSRBlockMatrixColMapOffd(P) = col_map_offd_P;
   	hypre_CSRBlockMatrixNumCols(P_offd) = num_cols_P_offd;
   } 

   /* use block version */
   hypre_GetCommPkgBlockRTFromCommPkgBlockA(P,A, fine_to_coarse_offd);


   *P_ptr = P;


   hypre_TFree(zero_block);
   hypre_TFree(identity_block);
   hypre_TFree(diagonal_block);
   hypre_TFree(sum_block);
   hypre_TFree(distribute_block);

   hypre_TFree(CF_marker_offd);
   hypre_TFree(int_buf_data);
   hypre_TFree(fine_to_coarse);
   hypre_TFree(fine_to_coarse_offd);
   hypre_TFree(coarse_counter);
   hypre_TFree(jj_count);
   hypre_TFree(jj_count_offd);

   if (num_procs > 1) hypre_CSRBlockMatrixDestroy(A_ext);

   return(0);  

}            
