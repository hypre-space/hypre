#include "_hypre_parcsr_ls.h"
#include "_hypre_utilities.h"
#include <cuda_runtime.h>

 __global__ void hypre_BoomerAMGBuildDirInterp_dev1( HYPRE_Int nr_of_rows, 
					HYPRE_Int* S_diag_i, HYPRE_Int* S_diag_j, 
					HYPRE_Int* S_offd_i, HYPRE_Int* S_offd_j,
					HYPRE_Int* CF_marker, HYPRE_Int* CF_marker_offd,
					HYPRE_Int num_functions, HYPRE_Int* dof_func, HYPRE_Int* dof_func_offd,
					HYPRE_Int* P_diag_i, HYPRE_Int* P_offd_i, HYPRE_Int* col_offd_S_to_A,
                                        HYPRE_Int* fine_to_coarse );

 __global__ void hypre_BoomerAMGBuildDirInterp_dev2( HYPRE_Int nr_of_rows, 
					HYPRE_Int* A_diag_i, HYPRE_Int* A_diag_j, HYPRE_Real* A_diag_data,
					HYPRE_Int* A_offd_i, HYPRE_Int* A_offd_j, HYPRE_Real* A_offd_data,
					HYPRE_Int* S_diag_i, HYPRE_Int* S_diag_j, 
					HYPRE_Int* S_offd_i, HYPRE_Int* S_offd_j,
					HYPRE_Int* CF_marker, HYPRE_Int* CF_marker_offd,
					HYPRE_Int num_functions, HYPRE_Int* dof_func, HYPRE_Int* dof_func_offd,
					HYPRE_Int* P_diag_i, HYPRE_Int* P_diag_j, HYPRE_Real* P_diag_data,
					HYPRE_Int* P_offd_i, HYPRE_Int* P_offd_j, HYPRE_Real* P_offd_data,
				        HYPRE_Int* col_offd_S_to_A, HYPRE_Int* fine_to_coarse );


/*---------------------------------------------------------------------------
 * hypre_BoomerAMGBuildDirInterp
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGBuildDirInterpDevice( hypre_ParCSRMatrix   *A,
                         HYPRE_Int                  *CF_marker,
                         hypre_ParCSRMatrix         *S,
                         HYPRE_BigInt               *num_cpts_global,
                         HYPRE_Int                   num_functions,
                         HYPRE_Int                  *dof_func,
                         HYPRE_Int                   debug_flag,
                         HYPRE_Real                  trunc_factor,
                         HYPRE_Int		     max_elmts,
                         HYPRE_Int 		    *col_offd_S_to_A,
                         hypre_ParCSRMatrix        **P_ptr)
{

   MPI_Comm 	      comm = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real      *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int       *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int       *A_diag_j = hypre_CSRMatrixJ(A_diag);

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Real      *A_offd_data = hypre_CSRMatrixData(A_offd);
   HYPRE_Int       *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int       *A_offd_j = hypre_CSRMatrixJ(A_offd);
   HYPRE_Int        num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);

   hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int       *S_diag_i = hypre_CSRMatrixI(S_diag);
   HYPRE_Int       *S_diag_j = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix *S_offd = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int       *S_offd_i = hypre_CSRMatrixI(S_offd);
   HYPRE_Int       *S_offd_j = hypre_CSRMatrixJ(S_offd);

   hypre_ParCSRMatrix *P;
   HYPRE_BigInt	      *col_map_offd_P;
   HYPRE_Int	      *tmp_map_offd = NULL;

   HYPRE_Int          *CF_marker_offd = NULL;
   HYPRE_Int          *dof_func_offd = NULL;

   hypre_CSRMatrix    *P_diag;
   hypre_CSRMatrix    *P_offd;

   HYPRE_Real      *P_diag_data;
   HYPRE_Int       *P_diag_i;
   HYPRE_Int       *P_diag_j;
   HYPRE_Real      *P_offd_data;
   HYPRE_Int       *P_offd_i;
   HYPRE_Int       *P_offd_j;

   HYPRE_Int        P_diag_size, P_offd_size;

   HYPRE_Int        jj_counter,jj_counter_offd;
   HYPRE_Int       *jj_count, *jj_count_offd;
   HYPRE_Int        jj_begin_row,jj_begin_row_offd;
   HYPRE_Int        jj_end_row,jj_end_row_offd;

   HYPRE_Int        start_indexing = 0; /* start indexing for P_data at 0 */

   HYPRE_Int        n_fine = hypre_CSRMatrixNumRows(A_diag);

   HYPRE_Int       *fine_to_coarse;
   HYPRE_Int       *coarse_counter;
   HYPRE_Int        coarse_shift;
   HYPRE_BigInt     total_global_cpts;
   HYPRE_Int        num_cols_P_offd;
   //HYPRE_BigInt     my_first_cpt;

   HYPRE_Int        i,i1;
   HYPRE_Int        j,jl,jj;
   HYPRE_Int        start;

   HYPRE_Real       diagonal;
   HYPRE_Real       sum_N_pos, sum_P_pos;
   HYPRE_Real       sum_N_neg, sum_P_neg;
   HYPRE_Real       alfa = 1.0;
   HYPRE_Real       beta = 1.0;

   HYPRE_Real       zero = 0.0;
   HYPRE_Real       one  = 1.0;

   HYPRE_Int        my_id;
   HYPRE_Int        num_procs;
   HYPRE_Int        num_threads;
   HYPRE_Int        num_sends;
   HYPRE_Int        index;
   HYPRE_Int        ns, ne, size, rest;
   HYPRE_Int       *int_buf_data;

   HYPRE_Real       wall_time;  /* for debugging instrumentation  */

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm,&my_id);
   num_threads = hypre_NumThreads();

#ifdef HYPRE_NO_GLOBAL_PARTITION
   //my_first_cpt = num_cpts_global[0];
   if (my_id == (num_procs -1)) total_global_cpts = num_cpts_global[1];
   hypre_MPI_Bcast(&total_global_cpts, 1, HYPRE_MPI_BIG_INT, num_procs-1, comm);
#else
   //my_first_cpt = num_cpts_global[my_id];
   total_global_cpts = num_cpts_global[num_procs];
#endif

   /*-------------------------------------------------------------------
    * Get the CF_marker data for the off-processor columns
    *-------------------------------------------------------------------*/

   if (debug_flag==4) wall_time = time_getWallclockSeconds();

//------------
// 1. Communicate CF_marker for offdiagonal par with other processors
//------------
   if (num_cols_A_offd) CF_marker_offd = hypre_CTAlloc(HYPRE_Int,  num_cols_A_offd, HYPRE_MEMORY_SHARED);
   if (num_functions > 1 && num_cols_A_offd)
	dof_func_offd = hypre_CTAlloc(HYPRE_Int,  num_cols_A_offd, HYPRE_MEMORY_SHARED);

   if (!comm_pkg)
   {
	hypre_MatvecCommPkgCreate(A);
	comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   int_buf_data = hypre_CTAlloc(HYPRE_Int,  hypre_ParCSRCommPkgSendMapStart(comm_pkg,
						num_sends), HYPRE_MEMORY_HOST);

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
//------------



   if (num_functions > 1)
   {
//------------
// 2. Communicate off diagonal part of dof_func with other processors
//------------
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
//------------


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

   //   coarse_counter = hypre_CTAlloc(HYPRE_Int,  num_threads, HYPRE_MEMORY_HOST);
   //   jj_count = hypre_CTAlloc(HYPRE_Int,  num_threads, HYPRE_MEMORY_HOST);
   //   jj_count_offd = hypre_CTAlloc(HYPRE_Int,  num_threads, HYPRE_MEMORY_HOST);

   fine_to_coarse = hypre_CTAlloc(HYPRE_Int,  n_fine, HYPRE_MEMORY_SHARED);
   //#ifdef HYPRE_USING_OPENMP
   //#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
   //#endif
   //   for (i = 0; i < n_fine; i++) fine_to_coarse[i] = -1;
   //
   //   jj_counter = start_indexing;
   //   jj_counter_offd = start_indexing;


   
   P_diag_i    = hypre_CTAlloc(HYPRE_Int,  n_fine+1, HYPRE_MEMORY_SHARED);
   P_offd_i    = hypre_CTAlloc(HYPRE_Int,  n_fine+1, HYPRE_MEMORY_SHARED);
   
//------------
// 1. Cuda kernel to figure out size of interpolation matrix, P, returns P_diag_i and P_offd_i
//------------   
   dim3 grid, block(32,1,1);
   grid.x = n_fine/block.x;
   if( n_fine % block.x != 0 )
      grid.x++;
   grid.y = 1;
   grid.z = 1;

   hypre_BoomerAMGBuildDirInterp_dev1<<<grid,block>>>( n_fine, S_diag_i, S_diag_j, S_offd_i, S_offd_j,
						       CF_marker, CF_marker_offd, num_functions, 
						       dof_func, dof_func_offd, P_diag_i, P_offd_i, 
						       col_offd_S_to_A, fine_to_coarse );
   cudaDeviceSynchronize();

   thrust::exclusive_scan( thrust::device, &P_diag_i[0], &P_diag_i[n_fine+1], &P_diag_i[0] );
   thrust::exclusive_scan( thrust::device, &P_offd_i[0], &P_offd_i[n_fine+1], &P_offd_i[0] );
   thrust::exclusive_scan( thrust::device, &fine_to_coarse[0], &fine_to_coarse[n_fine], &fine_to_coarse[0] );

   //   printf("fine_to_coarse=\n");
   //   for( i=0 ; i < n_fine ; i++ )
   //      printf("f to c %d %d \n",i,fine_to_coarse[i]);

   P_diag_size = P_diag_i[n_fine];
   P_diag_j    = hypre_CTAlloc(HYPRE_Int,  P_diag_size, HYPRE_MEMORY_SHARED);
   P_diag_data = hypre_CTAlloc(HYPRE_Real,  P_diag_size, HYPRE_MEMORY_SHARED);

   P_offd_size = P_offd_i[n_fine];
   P_offd_j    = hypre_CTAlloc(HYPRE_Int,  P_offd_size, HYPRE_MEMORY_SHARED);
   P_offd_data = hypre_CTAlloc(HYPRE_Real, P_offd_size, HYPRE_MEMORY_SHARED);


//------------
// 2. Cuda kernel to compute and fill in the elements of the interpolation matrix,
//    returns P_diag_j, P_diag_data, P_offd_j, and P_offd_data
//------------
   hypre_BoomerAMGBuildDirInterp_dev2<<<grid,block>>>( n_fine, A_diag_i, A_diag_j, A_diag_data, 
						 A_offd_i, A_offd_j, A_offd_data,
						 S_diag_i, S_diag_j, S_offd_i, S_offd_j,
						 CF_marker, CF_marker_offd,
						 num_functions, dof_func, dof_func_offd,
						 P_diag_i, P_diag_j, P_diag_data,
					         P_offd_i, P_offd_j, P_offd_data, 
						 col_offd_S_to_A, fine_to_coarse );
   cudaDeviceSynchronize();

   //   printf("after dev2 P_diag_i=\n");
   //   for( i=0 ; i < P_diag_i[n_fine] ; i++ )
   //      printf("P_diag_j %d %d \n",i,P_diag_j[i]);

   if( my_id==1 )
   {
      printf("after dev2 P_offd_i=\n");
      printf("P_offd_size = %d \n",P_offd_size);
      printf("P_diag_size = %d \n",P_diag_size);      
      //      for( i=0 ; i < P_offd_i[n_fine] ; i++ )
      //	 printf("P_offd_j %d %d \n",i,P_offd_j[i]);
      //	 printf("P_offd_j %d \n",i);	 
   }


//------------
// 3. Construct the result as a ParCSRMatrix
//------------
//   printf("pmatrix constructor, %d %d %d\n",total_global_cpts,P_diag_i[n_fine],P_offd_i[n_fine]);;
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
      HYPRE_Int *P_marker = hypre_CTAlloc(HYPRE_Int,  num_cols_A_offd, HYPRE_MEMORY_HOST);

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

      col_map_offd_P = hypre_CTAlloc(HYPRE_BigInt, num_cols_P_offd, HYPRE_MEMORY_HOST);
      tmp_map_offd = hypre_CTAlloc(HYPRE_Int, num_cols_P_offd, HYPRE_MEMORY_HOST);

      index = 0;
      for (i=0; i < num_cols_P_offd; i++)
      {
         while (P_marker[index]==0) index++;
         tmp_map_offd[i] = index++;
      }

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i=0; i < P_offd_size; i++)
	P_offd_j[i] = hypre_BinarySearch(tmp_map_offd,
					 P_offd_j[i],
					 num_cols_P_offd);
      hypre_TFree(P_marker, HYPRE_MEMORY_HOST);
   }

   for (i=0; i < n_fine; i++)
      if (CF_marker[i] == -3) CF_marker[i] = -1;

   if (num_cols_P_offd)
   {
   	hypre_ParCSRMatrixColMapOffd(P) = col_map_offd_P;
        hypre_CSRMatrixNumCols(P_offd) = num_cols_P_offd;
   }

   hypre_GetCommPkgRTFromCommPkgA(P, A, fine_to_coarse, tmp_map_offd);

   *P_ptr = P;

   hypre_TFree(CF_marker_offd, HYPRE_MEMORY_SHARED);
   hypre_TFree(dof_func_offd, HYPRE_MEMORY_SHARED);
   hypre_TFree(int_buf_data, HYPRE_MEMORY_HOST);
   hypre_TFree(fine_to_coarse, HYPRE_MEMORY_SHARED);
   hypre_TFree(tmp_map_offd, HYPRE_MEMORY_HOST);
   //   hypre_TFree(coarse_counter, HYPRE_MEMORY_HOST);
   //   hypre_TFree(jj_count, HYPRE_MEMORY_HOST);
   //   hypre_TFree(jj_count_offd, HYPRE_MEMORY_HOST);

   return hypre_error_flag;

}



 __global__ void hypre_BoomerAMGBuildDirInterp_dev1( HYPRE_Int nr_of_rows, 
				HYPRE_Int* S_diag_i, HYPRE_Int* S_diag_j, 
				HYPRE_Int* S_offd_i, HYPRE_Int* S_offd_j,
				HYPRE_Int* CF_marker, HYPRE_Int* CF_marker_offd,
				HYPRE_Int num_functions, HYPRE_Int* dof_func, HYPRE_Int* dof_func_offd,
				HYPRE_Int* P_diag_i, HYPRE_Int* P_offd_i, 
				HYPRE_Int* col_offd_S_to_A, HYPRE_Int* fine_to_coarse )
{
   /*-----------------------------------------------------------------------*/
   /* Determine size of interpolation matrix, P

      Input: nr_of_rows - Number of rows in matrix (local in processor)
             S_diag_i, S_diag_j - CSR representation of S_diag
             S_offd_i, S_offd_j - CSR representation of S_offd
	     num_function  - Number of degrees of freedom per grid point
	     dof_func      - vector over nonzero elements of A_diag, indicating the degree of freedom
	     dof_func_offd - vector over nonzero elements of A_offd, indicating the degree of freedom

      Output: P_diag_i      - P_diag_i vector for  P_diag
	      P_offd_i      - P_offd_i vector for  P_offd
    */
   /*-----------------------------------------------------------------------*/

   HYPRE_Int myid= threadIdx.x + blockIdx.x * blockDim.x, i, jj, i1, jPd, jPo;
   const HYPRE_Int nthreads = gridDim.x * blockDim.x;

   for( i = myid ; i < nr_of_rows ; i += nthreads )
   {
      /*--------------------------------------------------------------------
       *  If i is a C-point, interpolation is the identity.
       *--------------------------------------------------------------------*/
      jPd = jPo = 0;
      if (CF_marker[i] >= 0)
      {
	 jPd++;
	 fine_to_coarse[i] = 1;
      }
      else
      {
	 /*--------------------------------------------------------------------
	  *  If i is an F-point, interpolation is from the C-points that
	  *  strongly influence i.
	  *--------------------------------------------------------------------*/
         for (jj = S_diag_i[i]; jj < S_diag_i[i+1]; jj++)
         {
            i1 = S_diag_j[jj];
            if (CF_marker[i1] > 0 && (num_functions == 1 || (dof_func[i1]==dof_func[i])) )
	       jPd++;
         }
	 if (col_offd_S_to_A)
	 {
            for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj++)
            {
               i1 = col_offd_S_to_A[S_offd_j[jj]];
	       if (CF_marker_offd[i1] > 0 && ( num_functions == 1 || (dof_func_offd[i1]==dof_func[i])) )
		  jPo++;
	    }
	 }
	 else
	 {
	    for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj++)
	    {
	       i1 = S_offd_j[jj];
	       if (CF_marker_offd[i1] > 0 && ( num_functions ==1 || (dof_func_offd[i1]==dof_func[i] )) )
		     jPo++;
	    }
	 }
      }
      P_diag_i[i] = jPd;
      P_offd_i[i] = jPo;
   }
}

/*-----------------------------------------------------------------------*/
 __global__ void hypre_BoomerAMGBuildDirInterp_dev2( HYPRE_Int nr_of_rows, 
				HYPRE_Int* A_diag_i, HYPRE_Int* A_diag_j, HYPRE_Real* A_diag_data,
				HYPRE_Int* A_offd_i, HYPRE_Int* A_offd_j, HYPRE_Real* A_offd_data,
				HYPRE_Int* S_diag_i, HYPRE_Int* S_diag_j, 
				HYPRE_Int* S_offd_i, HYPRE_Int* S_offd_j,
				HYPRE_Int* CF_marker, HYPRE_Int* CF_marker_offd,
				HYPRE_Int num_functions, HYPRE_Int* dof_func, HYPRE_Int* dof_func_offd,
				HYPRE_Int* P_diag_i, HYPRE_Int* P_diag_j, HYPRE_Real* P_diag_data,
			        HYPRE_Int* P_offd_i, HYPRE_Int* P_offd_j, HYPRE_Real* P_offd_data,
			        HYPRE_Int* col_offd_S_to_A, HYPRE_Int* fine_to_coarse )
{
   /*-----------------------------------------------------------------------*/
   /* Compute interpolation matrix, P

      Input: nr_of_rows - Number of rows in matrix (local in processor)
             S_diag_i, S_diag_j - CSR representation of S_diag
             S_offd_i, S_offd_j - CSR representation of S_offd
	     num_function  - Number of degrees of freedom per grid point
	     dof_func      - vector over nonzero elements of A_diag, indicating the degree of freedom
	     dof_func_offd - vector over nonzero elements of A_offd, indicating the degree of freedom

      Output: P_diag_data      - P_diag_data vector for  P_diag
	      P_offd_data      - P_offd_data vector for  P_offd
   */
   /*-----------------------------------------------------------------------*/

   HYPRE_Int myid= threadIdx.x + blockIdx.x * blockDim.x, i, ind, indp, inds, i1;
   HYPRE_Real diagonal, sum_N_pos, sum_N_neg, sum_P_pos, sum_P_neg, alfa, beta;

   const HYPRE_Int nthreads = gridDim.x * blockDim.x;

   for( i = myid ; i < nr_of_rows ; i += nthreads )
   {
      if( CF_marker[i] > 0 )
      {
      /*--------------------------------------------------------------------
       *  If i is a c-point, interpolation is the identity.
       *--------------------------------------------------------------------*/
	 ind = P_diag_i[i];
	 P_diag_j[ind]    = fine_to_coarse[i];
	 P_diag_data[ind] = 1.0;
      }
      else
      {
      /*--------------------------------------------------------------------
       *  Point is f-point, use direct interpolation 
       *--------------------------------------------------------------------*/
	 sum_N_pos = 0;
	 sum_N_neg = 0;
	 sum_P_pos = 0;
	 sum_P_neg = 0;
	 inds=S_diag_i[i];
	 indp=P_diag_i[i];
	 diagonal = A_diag_data[A_diag_i[i]];

	 /* The loops below assume that the sparsity structure of S was obtained  */
	 /* by removing elements from the sparsity structure of A, but that no  */
         /* reordering of S (or A) has been done */

	 for( ind=A_diag_i[i]+1; ind < A_diag_i[i+1] ; ind++ )
	 {
            i1 = A_diag_j[ind];
	    if (num_functions == 1 || dof_func[i1] == dof_func[i])
	    {
	       if (A_diag_data[ind] > 0)
	          sum_N_pos += A_diag_data[ind];
	       else
	          sum_N_neg += A_diag_data[ind];
	    }
	    if( inds < S_diag_i[i+1] && i1==S_diag_j[inds] )
	    {
	       /* Element is in both A and S */
	       if (CF_marker[i1] > 0 && ( num_functions==1 || dof_func[i1]==dof_func[i]) )
	       {
		  P_diag_data[indp] = A_diag_data[ind];
		  P_diag_j[indp++] = fine_to_coarse[i1];
		  if( A_diag_data[ind] > 0 )
		     sum_P_pos += A_diag_data[ind];
		  else
		     sum_P_neg += A_diag_data[ind];		     
	       }
	       inds++;
	    }
	 }
	 inds=S_offd_i[i];
	 indp=P_offd_i[i];
	 for( ind=A_offd_i[i]+1; ind < A_offd_i[i+1] ; ind++ )
	 {
            i1 = A_offd_j[ind];
	    if (num_functions == 1 || dof_func_offd[i1] == dof_func[i])
	    {
	       if (A_offd_data[ind] > 0)
	          sum_N_pos += A_offd_data[ind];
	       else
	          sum_N_neg += A_offd_data[ind];
	    }
	    if( col_offd_S_to_A )
	       i1 = col_offd_S_to_A[S_offd_j[inds]];
	    if( inds < S_offd_i[i+1] && i1==S_offd_j[inds] )
	    {
	       /* Element is in both A and S */
	       if (CF_marker_offd[i1] > 0 && ( num_functions == 1 || dof_func_offd[i1]==dof_func[i] ) )
	       {
		  P_offd_data[indp] = A_offd_data[ind];
		  P_offd_j[indp++] = i1;
		  if( A_offd_data[ind] > 0 )
		     sum_P_pos += A_offd_data[ind];
		  else
		     sum_P_neg += A_offd_data[ind];		     
	       }
	       inds++;
	    }
	 }
	 if (sum_P_neg) alfa = sum_N_neg/(sum_P_neg*diagonal);
         if (sum_P_pos) beta = sum_N_pos/(sum_P_pos*diagonal);
	 
	 for( indp=P_diag_i[i]; indp < P_diag_i[i+1] ; indp++ )
	 {
            if (P_diag_data[indp]> 0)
               P_diag_data[indp] *= -beta;
            else
               P_diag_data[indp] *= -alfa;
	 }
	 for( indp=P_offd_i[i]; indp < P_offd_i[i+1] ; indp++ )
	 {
            if (P_offd_data[indp]> 0)
               P_offd_data[indp] *= -beta;
            else
               P_offd_data[indp] *= -alfa;
	 }
      }
   }
}
