#include "_hypre_parcsr_ls.h"

// TODO
#if 0 // comment out for now to pass regression tests

//nvlink warning : Stack size for entry function '_Z42hypre_BoomerAMGInterpTruncationDevice_dev3iPiS_PdS_S_S0_S_S_i' cannot be statically determined ??? What's wrong???

HYPRE_Int
hypre_BoomerAMGInterpTruncationDevice( hypre_ParCSRMatrix *P,
				       HYPRE_Real trunc_factor,
				       HYPRE_Int max_elmts);

 __global__ void hypre_BoomerAMGBuildDirInterp_dev1( HYPRE_Int nr_of_rows,
					HYPRE_Int* S_diag_i, HYPRE_Int* S_diag_j,
					HYPRE_Int* S_offd_i, HYPRE_Int* S_offd_j,
					HYPRE_Int* A_offd_i, HYPRE_Int* A_offd_j,
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


__global__ void hypre_BoomerAMGBuildDirInterp_dev3( HYPRE_Int P_offd_size,
						    HYPRE_Int* P_offd_j,
						    HYPRE_Int* P_marker );

__global__ void hypre_BoomerAMGBuildDirInterp_dev4( HYPRE_Int num_cols_A_offd,
						    HYPRE_Int* P_marker,
						    HYPRE_Int* tmp_map_offd );

__global__ void hypre_BoomerAMGBuildDirInterp_dev5( HYPRE_Int P_offd_size,
						    HYPRE_Int* P_offd_j,
						    HYPRE_Int* P_marker );

__global__ void hypre_BoomerAMGInterpTruncationDevice_dev1( HYPRE_Int num_rows_P,
							    HYPRE_Int* P_diag_i,
							    HYPRE_Int* P_diag_j,
							    HYPRE_Real* P_diag_data,
							    HYPRE_Int* P_offd_i,
							    HYPRE_Int* P_offd_j,
							    HYPRE_Real* P_offd_data,
							    HYPRE_Int* P_aux_diag_i,
							    HYPRE_Int* P_aux_offd_i,
							    HYPRE_Real trunc_factor );

__global__ void hypre_BoomerAMGInterpTruncationDevice_dev2( HYPRE_Int  num_rows_P,
							    HYPRE_Int* P_diag_i,
							    HYPRE_Int* P_offd_i,
							    HYPRE_Int* P_aux_diag_i,
							    HYPRE_Int* P_aux_offd_i );

__global__ void hypre_BoomerAMGInterpTruncationDevice_dev3( HYPRE_Int   num_rows_P,
							    HYPRE_Int*  P_diag_i,
							    HYPRE_Int*  P_diag_j,
							    HYPRE_Real* P_diag_data,
							    HYPRE_Int*  P_offd_i,
							    HYPRE_Int*  P_offd_j,
							    HYPRE_Real* P_offd_data,
							    HYPRE_Int*  P_aux_diag_i,
							    HYPRE_Int*  P_aux_offd_i,
							    HYPRE_Int   max_elements );

__global__ void hypre_BoomerAMGInterpTruncationDevice_dev4( HYPRE_Int   num_rows_P,
							    HYPRE_Int*  P_diag_i,
							    HYPRE_Int*  P_diag_j,
							    HYPRE_Real* P_diag_data,
							    HYPRE_Int*  P_diag_i_new,
							    HYPRE_Int*  P_diag_j_new,
							    HYPRE_Real* P_diag_data_new,
							    HYPRE_Int*  P_offd_i,
							    HYPRE_Int*  P_offd_j,
							    HYPRE_Real* P_offd_data,
							    HYPRE_Int*  P_offd_i_new,
							    HYPRE_Int*  P_offd_j_new,
							    HYPRE_Real* P_offd_data_new );

__device__ void hypre_qsort2abs_dev( HYPRE_Int *v,
				     HYPRE_Real *w,
				     HYPRE_Int  left,
				     HYPRE_Int  right );

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
   //   HYPRE_BigInt	   *col_map_offd_A = hypre_ParCSRMatrixColMapOffd(A);

   hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int       *S_diag_i = hypre_CSRMatrixI(S_diag);
   HYPRE_Int       *S_diag_j = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix *S_offd = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int       *S_offd_i = hypre_CSRMatrixI(S_offd);
   HYPRE_Int       *S_offd_j = hypre_CSRMatrixJ(S_offd);

   hypre_ParCSRMatrix *P;
   HYPRE_BigInt	      *col_map_offd_P;
   HYPRE_Int	      *tmp_map_offd = NULL;

   HYPRE_Int          *CF_marker_host = NULL;
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

   HYPRE_Int        n_fine = hypre_CSRMatrixNumRows(A_diag);

   HYPRE_Int       *fine_to_coarse;
   HYPRE_BigInt     total_global_cpts;
   HYPRE_Int        num_cols_P_offd;

   HYPRE_Int        i;
   HYPRE_Int        j;
   HYPRE_Int        start;

   HYPRE_Int        my_id;
   HYPRE_Int        num_procs;
   HYPRE_Int        num_sends;
   HYPRE_Int        index;
   HYPRE_Int       *int_buf_data;
   HYPRE_Int        limit = 1048576;

   HYPRE_Real       wall_time;  /* for debugging instrumentation  */

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm,&my_id);

#ifdef HYPRE_NO_GLOBAL_PARTITION
   if (my_id == (num_procs -1))
      total_global_cpts = num_cpts_global[1];
   hypre_MPI_Bcast( &total_global_cpts, 1, HYPRE_MPI_BIG_INT, num_procs-1, comm);
#else
   total_global_cpts = num_cpts_global[num_procs];
#endif

   if (!comm_pkg)
   {
	hypre_MatvecCommPkgCreate(A);
	comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }
   if (debug_flag==4) wall_time = time_getWallclockSeconds();

/* 0. Assume CF_marker has been allocated in device memory */
   CF_marker_host = hypre_CTAlloc(HYPRE_Int,  n_fine, HYPRE_MEMORY_HOST);
   hypre_TMemcpy( CF_marker_host, CF_marker, HYPRE_Int, n_fine, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE );

/* 1. Communicate CF_marker to/from other processors */
   if (num_cols_A_offd)
      CF_marker_offd = hypre_CTAlloc(HYPRE_Int,  num_cols_A_offd, HYPRE_MEMORY_SHARED);

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   int_buf_data = hypre_CTAlloc(HYPRE_Int,  hypre_ParCSRCommPkgSendMapStart(comm_pkg,
						num_sends), HYPRE_MEMORY_HOST);
   index = 0;
   for (i = 0; i < num_sends; i++)
   {
	start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
	for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
		int_buf_data[index++]
		 = CF_marker_host[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
   }

   comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data, CF_marker_offd);
   hypre_ParCSRCommHandleDestroy(comm_handle);
   hypre_TFree(CF_marker_host, HYPRE_MEMORY_HOST);

   if (num_functions > 1)
   {
 /* 2. Communicate dof_func to/from other processors */
      if (num_cols_A_offd > 0)
	 dof_func_offd = hypre_CTAlloc(HYPRE_Int,  num_cols_A_offd, HYPRE_MEMORY_SHARED);

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
	 start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
	 for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
		int_buf_data[index++]
		 = dof_func[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
      }
      comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data, dof_func_offd);
      hypre_ParCSRCommHandleDestroy(comm_handle);
   }


   if (debug_flag==4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d     Interp: Comm 1 CF_marker =    %f\n",
                    my_id, wall_time);
      fflush(NULL);
   }

/* 3. Figure out the size of the interpolation matrix, P, i.e., compute P_diag_i and P_offd_i */
/*    Also, compute fine_to_coarse array: When i is a coarse point, fine_to_coarse[i] will hold a  */
/*    corresponding coarse point index in the range 0..n_coarse-1 */
   P_diag_i    = hypre_CTAlloc(HYPRE_Int,  n_fine+1, HYPRE_MEMORY_SHARED);
   P_offd_i    = hypre_CTAlloc(HYPRE_Int,  n_fine+1, HYPRE_MEMORY_SHARED);
   fine_to_coarse = hypre_CTAlloc(HYPRE_Int,  n_fine, HYPRE_MEMORY_SHARED);

   dim3 grid, block(32,1,1);
   grid.x = n_fine/block.x;
   if( n_fine % block.x != 0 )
      grid.x++;
   if( grid.x > limit )
      grid.x = limit;
   grid.y = 1;
   grid.z = 1;
   hypre_BoomerAMGBuildDirInterp_dev1<<<grid,block>>>( n_fine, S_diag_i, S_diag_j, S_offd_i, S_offd_j,
                                                       A_offd_i, A_offd_j,
						       CF_marker, CF_marker_offd, num_functions,
						       dof_func, dof_func_offd, P_diag_i, P_offd_i,
						       col_offd_S_to_A, fine_to_coarse );

 /* The scans will transform P_diag_i and P_offd_i to the CSR I-vectors */
   thrust::exclusive_scan( thrust::device, &P_diag_i[0], &P_diag_i[n_fine+1], &P_diag_i[0] );
   thrust::exclusive_scan( thrust::device, &P_offd_i[0], &P_offd_i[n_fine+1], &P_offd_i[0] );
/* The scan will make fine_to_coarse[i] for i a coarse point hold a coarse point index in the range from 0 to n_coarse-1 */
   thrust::exclusive_scan( thrust::device, &fine_to_coarse[0], &fine_to_coarse[n_fine], &fine_to_coarse[0] );

/* 4. Compute the CSR arrays P_diag_j, P_diag_data, P_offd_j, and P_offd_data */
/*    P_diag_i and P_offd_i are now known, first allocate the remaining CSR arrays of P */

   P_diag_size = P_diag_i[n_fine];
   P_diag_j    = hypre_CTAlloc(HYPRE_Int,  P_diag_size, HYPRE_MEMORY_SHARED);
   P_diag_data = hypre_CTAlloc(HYPRE_Real, P_diag_size, HYPRE_MEMORY_SHARED);

   P_offd_size = P_offd_i[n_fine];
   P_offd_j    = hypre_CTAlloc(HYPRE_Int,  P_offd_size, HYPRE_MEMORY_SHARED);
   P_offd_data = hypre_CTAlloc(HYPRE_Real, P_offd_size, HYPRE_MEMORY_SHARED);


   hypre_BoomerAMGBuildDirInterp_dev2<<<grid,block>>>( n_fine, A_diag_i, A_diag_j, A_diag_data,
						 A_offd_i, A_offd_j, A_offd_data,
						 S_diag_i, S_diag_j, S_offd_i, S_offd_j,
						 CF_marker, CF_marker_offd,
						 num_functions, dof_func, dof_func_offd,
						 P_diag_i, P_diag_j, P_diag_data,
					         P_offd_i, P_offd_j, P_offd_data,
						 col_offd_S_to_A, fine_to_coarse );
   cudaDeviceSynchronize();

/* 5. Construct the result as a ParCSRMatrix. At this point, P's column indices */
/*    are defined with A's enumeration of columns */

   P = hypre_ParCSRMatrixCreate(comm,
                                hypre_ParCSRMatrixGlobalNumRows(A),
                                total_global_cpts,
                                hypre_ParCSRMatrixColStarts(A),
                                num_cpts_global,
                                0,
                                P_diag_size,
                                P_offd_size);

   P_diag = hypre_ParCSRMatrixDiag(P);
   hypre_CSRMatrixData(P_diag) = P_diag_data;
   hypre_CSRMatrixI(P_diag) = P_diag_i;
   hypre_CSRMatrixJ(P_diag) = P_diag_j;
   P_offd = hypre_ParCSRMatrixOffd(P);
   hypre_CSRMatrixData(P_offd) = P_offd_data;
   hypre_CSRMatrixI(P_offd) = P_offd_i;
   hypre_CSRMatrixJ(P_offd) = P_offd_j;
   hypre_ParCSRMatrixOwnsRowStarts(P) = 0;


/* 6. Compress P, removing coefficients smaller than trunc_factor * Max, and */
/*    make sure no row has more than max_elmts elements */

   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      hypre_BoomerAMGInterpTruncationDevice(P, trunc_factor, max_elmts);
      P_diag_data = hypre_CSRMatrixData(P_diag);
      P_diag_i = hypre_CSRMatrixI(P_diag);
      P_diag_j = hypre_CSRMatrixJ(P_diag);
      P_offd_data = hypre_CSRMatrixData(P_offd);
      P_offd_i = hypre_CSRMatrixI(P_offd);
      P_offd_j = hypre_CSRMatrixJ(P_offd);
      P_offd_size = P_offd_i[n_fine];
   }

/* 7. Translate P_offd's column indices from the values inherited from A_offd to a 0,1,2,3,... enumeration, */
/*    and construct the col_map array that translates these into the global 0..c-1 enumeration */
/*    (P is of size m x c) */

   num_cols_P_offd = 0;
   if (P_offd_size)
   {
      /* Array P_marker has length equal to the number of A's offd columns+1, and will */
      /* store a translation code from A_offd's local column numbers to P_offd's local column numbers */
      /* Example: if A_offd has 6 columns, locally 0,1,..,5, and points 1 and 4 are coarse points, then
         P_marker=[0,1,0,0,1,0,0], */

      /* First,  set P_marker[i] to 1 if A's column i is also present in P, otherwise P_marker[i] is 0 */
      HYPRE_Int *P_marker = hypre_CTAlloc(HYPRE_Int,  num_cols_A_offd+1, HYPRE_MEMORY_DEVICE);
      hypre_BoomerAMGBuildDirInterp_dev5<<<grid,block>>>( P_offd_size, P_offd_j, P_marker );

      /* Secondly, the sum over P_marker gives the number of different columns in P's offd part */
      num_cols_P_offd = thrust::reduce(thrust::device,&P_marker[0],&P_marker[num_cols_A_offd]);

      /* Because P's columns correspond to P_marker[i]=1 (and =0 otherwise), the scan below will return  */
      /* an enumeration of P's columns 0,1,... at the corresponding locations in P_marker. */
      /* P_marker[num_cols_A_offd] will contain num_cols_P_offd, so sum reduction above could  */
      /* have been replaced by reading the last element of P_marker. */
      thrust::exclusive_scan( thrust::device, &P_marker[0], &P_marker[num_cols_A_offd+1], &P_marker[0] );
      /* Example: P_marker becomes [0,0,1,1,1,2] so that P_marker[1]=0, P_marker[4]=1  */

      /* Do the re-enumeration, P_offd_j are mapped, using P_marker as map  */
      hypre_BoomerAMGBuildDirInterp_dev3<<<grid,block>>>( P_offd_size, P_offd_j, P_marker );

      /* Create and define array tmp_map_offd. This array is the inverse of the P_marker mapping, */
      /* Example: num_cols_P_offd=2, tmp_map_offd[0] = 1, tmp_map_offd[1]=4  */
      tmp_map_offd   = hypre_CTAlloc(HYPRE_Int, num_cols_P_offd, HYPRE_MEMORY_SHARED);
      grid.x = num_cols_A_offd/block.x;
      if( num_cols_A_offd % block.x != 0 )
	 grid.x++;
      if( grid.x > limit )
	 grid.x = limit;

      hypre_BoomerAMGBuildDirInterp_dev4<<<grid,block>>>( num_cols_A_offd, P_marker, tmp_map_offd );

      if (num_cols_P_offd)
      {
	 col_map_offd_P = hypre_CTAlloc(HYPRE_BigInt, num_cols_P_offd, HYPRE_MEMORY_SHARED);
	 hypre_ParCSRMatrixColMapOffd(P) = col_map_offd_P;
	 hypre_CSRMatrixNumCols(P_offd) = num_cols_P_offd;
      }

      cudaDeviceSynchronize();      
      hypre_TFree(P_marker, HYPRE_MEMORY_DEVICE);
   }

   /* Not sure what this is for, moved it to Cuda kernel _dev2 */
   /*   for (i=0; i < n_fine; i++)
	if (CF_marker[i] == -3) CF_marker[i] = -1; */


/* 8. P_offd_j now has a 0,1,2,3... local column index enumeration. */
/*    tmp_map_offd contains the index mapping from P's offd local columns to A's offd local columns.*/
/*    Below routine is in parcsr_ls/par_rap_communication.c. It sets col_map_offd in P, */
/*    comm_pkg in P, and perhaps more members of P ??? */
   hypre_GetCommPkgRTFromCommPkgA(P, A, fine_to_coarse, tmp_map_offd);

   *P_ptr = P;

   hypre_TFree(CF_marker_offd, HYPRE_MEMORY_SHARED);
   hypre_TFree(dof_func_offd, HYPRE_MEMORY_SHARED);
   hypre_TFree(int_buf_data, HYPRE_MEMORY_HOST);
   hypre_TFree(fine_to_coarse, HYPRE_MEMORY_SHARED);
   hypre_TFree(tmp_map_offd, HYPRE_MEMORY_SHARED);

   return hypre_error_flag;
}


/*-----------------------------------------------------------------------*/
 __global__ void hypre_BoomerAMGBuildDirInterp_dev1( HYPRE_Int nr_of_rows,
				HYPRE_Int* S_diag_i, HYPRE_Int* S_diag_j,
				HYPRE_Int* S_offd_i, HYPRE_Int* S_offd_j,
				HYPRE_Int* A_offd_i, HYPRE_Int* A_offd_j,							                                         HYPRE_Int* CF_marker, HYPRE_Int* CF_marker_offd,
				HYPRE_Int num_functions, HYPRE_Int* dof_func, HYPRE_Int* dof_func_offd,
				HYPRE_Int* P_diag_i, HYPRE_Int* P_offd_i,
				HYPRE_Int* col_offd_S_to_A, HYPRE_Int* fine_to_coarse )
{
   /*-----------------------------------------------------------------------*/
   /* Determine size of interpolation matrix, P

      If A is of size m x m, then P will be of size m x c where c is the
      number of coarse points.

      It is assumed that S have the same global column enumeration as A

      Input: nr_of_rows         - Number of rows in matrix (local in processor)
             S_diag_i, S_diag_j - CSR representation of S_diag
             S_offd_i, S_offd_j - CSR representation of S_offd
	     num_function  - Number of degrees of freedom per grid point
	     dof_func      - vector of length nr_of_rows, indicating the degree of freedom of vector element.
	     dof_func_offd - vector over nonzero elements of A_offd, indicating the degree of freedom.

      Output: P_diag_i       - Vector where P_diag_i[i] holds the number of non-zero elements of P_diag on row i.
	      P_offd_i       - Vector where P_offd_i[i] holds the number of non-zero elements of P_offd on row i.
              fine_to_coarse - Vector of length nr_of_rows-1.
                               fine_to_coarse[i] is set to 1 if i is a coarse pt.
                               Eventually, fine_to_coarse[j] will map A's column j
                               to a re-enumerated column index in matrix P.
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
	       i1 = S_offd_j[jj]; /* CF_marker_offd[i1] requires i1 to be in A's column enumeration */
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
             A_diag_i, A_diag_j, A_diag_data - CSR representation of A_diag
             A_offd_i, A_offd_j, A_offd_data - CSR representation of A_offd
             S_diag_i, S_diag_j - CSR representation of S_diag
             S_offd_i, S_offd_j - CSR representation of S_offd
	     CF_marker          - Coarse/Fine flags for indices (rows) in this processor
	     CF_marker_offd     - Coarse/Fine flags for indices (rows) not in this processor
	     num_function  - Number of degrees of freedom per grid point
	     dof_func      - vector over nonzero elements of A_diag, indicating the degree of freedom
	     dof_func_offd - vector over nonzero elements of A_offd, indicating the degree of freedom
             fine_to_coarse - Vector of length nr_of_rows-1.

      Output: P_diag_j         - Column indices in CSR representation of P_diag
              P_diag_data      - Matrix elements in CSR representation of P_diag
              P_offd_j         - Column indices in CSR representation of P_offd
	      P_offd_data      - Matrix elements in CSR representation of P_diag
   */
   /*-----------------------------------------------------------------------*/

   HYPRE_Int myid= threadIdx.x + blockIdx.x * blockDim.x, i, ind, indp, inds, i1;
   HYPRE_Real diagonal, sum_N_pos, sum_N_neg, sum_P_pos, sum_P_neg, alfa, beta, aval;
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
	 sum_N_pos = sum_N_neg = sum_P_pos = sum_P_neg = 0;
	 inds=S_diag_i[i];
	 indp=P_diag_i[i];
	 diagonal = A_diag_data[A_diag_i[i]];

	 /* The loops below assume that the sparsity structure of S was obtained  */
	 /* by removing elements from the sparsity structure of A, but that no  */
         /* reordering of S (or A) has been done */

	 for( ind=A_diag_i[i]+1; ind < A_diag_i[i+1] ; ind++ )
	 {
            i1   = A_diag_j[ind];
	    aval = A_diag_data[ind];
	    if (num_functions == 1 || dof_func[i1] == dof_func[i])
	    {
	       if( aval > 0 )
		  sum_N_pos += aval;
	       else
		  sum_N_neg += aval;
	       //	       if (A_diag_data[ind] > 0)
	       //	          sum_N_pos += A_diag_data[ind];
	       //	       else
	       //	          sum_N_neg += A_diag_data[ind];
	    }
	    if( inds < S_diag_i[i+1] && i1==S_diag_j[inds] )
	    {
	       /* Element is in both A and S */
	       if (CF_marker[i1] > 0 && ( num_functions==1 || dof_func[i1]==dof_func[i]) )
	       {
		  //		  P_diag_data[indp] = A_diag_data[ind];
		  P_diag_data[indp] = aval;
		  P_diag_j[indp++]  = fine_to_coarse[i1];
		  //		  if( A_diag_data[ind] > 0 )
		  //		     sum_P_pos += A_diag_data[ind];
		  //		  else
		  //		     sum_P_neg += A_diag_data[ind];
		  if( aval > 0 )
		     sum_P_pos += aval;
		  else
		     sum_P_neg += aval;
	       }
	       inds++;
	    }
	 }
	 inds=S_offd_i[i];
	 indp=P_offd_i[i];
	 for( ind=A_offd_i[i]; ind < A_offd_i[i+1] ; ind++ )
	 {
            i1   = A_offd_j[ind];
	    aval = A_offd_data[ind];
	    if (num_functions == 1 || dof_func_offd[i1] == dof_func[i])
	    {
	       if (aval > 0)
	          sum_N_pos += aval;
	       else
	          sum_N_neg += aval;
	    }
	    if( col_offd_S_to_A )
	       i1 = col_offd_S_to_A[S_offd_j[inds]];
	    if( inds < S_offd_i[i+1] && i1==S_offd_j[inds] )
	    {
	       /* Element is in both A and S */
	       if (CF_marker_offd[i1] > 0 && ( num_functions == 1 || dof_func_offd[i1]==dof_func[i] ) )
	       {
		  P_offd_data[indp] = aval;
		  P_offd_j[indp++]  = i1;
		  if( aval > 0 )
		     sum_P_pos += aval;
		  else
		     sum_P_neg += aval;
	       }
	       inds++;
	    }
	 }
	 alfa=beta=1.0;
	 if (sum_P_neg) alfa = sum_N_neg/(sum_P_neg*diagonal);
         if (sum_P_pos) beta = sum_N_pos/(sum_P_pos*diagonal);

	 for( indp=P_diag_i[i]; indp < P_diag_i[i+1] ; indp++ )
	 {
	    P_diag_data[indp] *= (P_diag_data[indp]>0)*(alfa-beta)-alfa;
	    //            if (P_diag_data[indp]> 0)
	    //               P_diag_data[indp] *= -beta;
	    //            else
	    //               P_diag_data[indp] *= -alfa;
	 }
	 for( indp=P_offd_i[i]; indp < P_offd_i[i+1] ; indp++ )
	 {
	    P_offd_data[indp] *= (P_offd_data[indp]>0)*(alfa-beta)-alfa;
	    //            if (P_offd_data[indp]> 0)
	    //               P_offd_data[indp] *= -beta;
	    //            else
	    //               P_offd_data[indp] *= -alfa;
	 }
	 if( CF_marker[i] == -3 )
	    CF_marker[i] = -1;
      }
   }
}

/*-----------------------------------------------------------------------*/
__global__ void hypre_BoomerAMGBuildDirInterp_dev3( HYPRE_Int P_offd_size,
						    HYPRE_Int* P_offd_j,
						    HYPRE_Int* P_marker )
/*
   Re-enumerate the columns of P_offd according to the mapping given in P_marker

 */
{
   HYPRE_Int myid= threadIdx.x + blockIdx.x * blockDim.x, i;
   const HYPRE_Int nthreads = gridDim.x * blockDim.x;

   for( i = myid ; i < P_offd_size ; i += nthreads )
       P_offd_j[i] = P_marker[P_offd_j[i]];
}


/*-----------------------------------------------------------------------*/
__global__ void hypre_BoomerAMGBuildDirInterp_dev4( HYPRE_Int num_cols_A_offd,
						    HYPRE_Int* P_marker,
						    HYPRE_Int* tmp_map_offd )
{
   /* Construct array tmp_map_offd

      Note: This is an inefficient kernel, its only purpose is to make it
            possible to keep the arrays on device */
   HYPRE_Int myid= threadIdx.x + blockIdx.x * blockDim.x, i;
   const HYPRE_Int nthreads = gridDim.x * blockDim.x;

   for( i = myid ; i < num_cols_A_offd ; i += nthreads )
   {
      if( P_marker[i] < P_marker[i+1] )
	 tmp_map_offd[P_marker[i]] = i;
   }
}

/*-----------------------------------------------------------------------*/
__global__ void hypre_BoomerAMGBuildDirInterp_dev5( HYPRE_Int P_offd_size,
						    HYPRE_Int* P_offd_j,
						    HYPRE_Int* P_marker )
/*
     set P_marker[i] to 1 if A's column i is also present in P, otherwise P_marker[i] is 0 
 */
{
   HYPRE_Int myid= threadIdx.x + blockIdx.x * blockDim.x, i;
   const HYPRE_Int nthreads = gridDim.x * blockDim.x;
   for( i = myid ; i < P_offd_size ; i += nthreads )
   {
      atomicOr(&P_marker[P_offd_j[i]],1);
   }
}


/*-----------------------------------------------------------------------*/
/*
   typedef thrust::zip_iterator<thrust::tuple<thrust::device_vector<HYPRE_Int>::iterator,thrust::device_vector<HYPRE_Int>::iterator > > ZIvec2Iterator;
   struct compare_tuple
   {
      template<typename Tuple>
      __host__ __device__
      bool operator()(Tuple lhs, Tuple rhs)
      {
         return thrust::get<0>(lhs)+thrust::get<1>(lhs) < thrust::get<0>(rhs)+thrust::get<1>(rhs);
      }
   };
*/

/*-----------------------------------------------------------------------*/
HYPRE_Int
hypre_BoomerAMGInterpTruncationDevice( hypre_ParCSRMatrix *P,
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
   HYPRE_Int* P_aux_diag_i=NULL;
   HYPRE_Int* P_aux_offd_i=NULL;
   HYPRE_Int* nel_per_row;

   HYPRE_Int n_fine = hypre_CSRMatrixNumRows(P_diag);
   HYPRE_Int P_diag_size;
   HYPRE_Int P_offd_size;
   HYPRE_Int mx_row;
   HYPRE_Int limit=1048576;/* arbitrarily choosen limit on CUDA grid size, most devices seem to have 2^31 as grid size limit */
   bool truncated = false;

   dim3 grid, block(32,1,1);
   grid.x = n_fine/block.x;
   if( n_fine % block.x != 0 )
      grid.x++;
   if( grid.x > limit )
      grid.x = limit;
   grid.y = 1;
   grid.z = 1;

   if( 0.0 < trunc_factor && trunc_factor < 1.0 )
   {
   /* truncate with trunc_factor, return number of remaining elements/row in P_aux_diag_i and P_aux_offd_i */
      P_aux_diag_i = hypre_CTAlloc(HYPRE_Int, n_fine+1, HYPRE_MEMORY_SHARED);
      P_aux_offd_i = hypre_CTAlloc(HYPRE_Int, n_fine+1, HYPRE_MEMORY_SHARED);
      hypre_BoomerAMGInterpTruncationDevice_dev1<<<grid,block>>>( n_fine, P_diag_i, P_diag_j, P_diag_data,
							 P_offd_i, P_offd_j, P_offd_data,
							 P_aux_diag_i, P_aux_offd_i, trunc_factor);
      truncated = true;
   }

   if( max_elmts > 0 )
   {
      if( !truncated )
      {
         /* If not previously truncated, set up P_aux_diag_i and P_aux_offd_i with full number of elements/row */
         P_aux_diag_i = hypre_CTAlloc(HYPRE_Int, n_fine+1, HYPRE_MEMORY_SHARED);
         P_aux_offd_i = hypre_CTAlloc(HYPRE_Int, n_fine+1, HYPRE_MEMORY_SHARED);
         hypre_BoomerAMGInterpTruncationDevice_dev2<<<grid,block >>>( n_fine, P_diag_i, P_offd_i, P_aux_diag_i, P_aux_offd_i );
      }
      nel_per_row = hypre_CTAlloc(HYPRE_Int, n_fine, HYPRE_MEMORY_DEVICE);
      thrust::transform(thrust::device,&P_aux_diag_i[0],&P_aux_diag_i[n_fine],&P_aux_offd_i[0],&nel_per_row[0],thrust::plus<HYPRE_Int>() );
      mx_row = thrust::reduce(thrust::device,&nel_per_row[0],&nel_per_row[n_fine],0,thrust::maximum<HYPRE_Int>());
      hypre_TFree(nel_per_row,HYPRE_MEMORY_DEVICE);

      /* Use zip_iterator to avoid creating help array nel_per_row */
      /*
      ZIvec2Iterator i = thrust::max_element(thrust::device,thrust::make_zip_iterator(&P_aux_diag_i[0],&P_aux_offd_i[0]),
					     thrust::make_zip_iterator(&P_aux_diag_i[n_fine],&P_aux_offd_i[n_fine],compare_tuple() ));
      mx_row = thrust::get<0>(*i)+thrust::get<1>(*i);
       */
      if( mx_row > max_elmts )
      {
       /* Truncate with respect to maximum number of elements per row */
	 hypre_BoomerAMGInterpTruncationDevice_dev3<<<grid,block>>>( n_fine, P_diag_i, P_diag_j, P_diag_data,
							    P_offd_i, P_offd_j, P_offd_data, P_aux_diag_i,
							    P_aux_offd_i, max_elmts );
	 truncated = true;
      }
   }

   if( truncated )
   {
    /* Matrix has been truncated, reshuffle it into shorter arrays */
      thrust::exclusive_scan(thrust::device,&P_aux_diag_i[0],&P_aux_diag_i[n_fine+1],&P_aux_diag_i[0]);
      P_diag_size = P_aux_diag_i[n_fine];
      thrust::exclusive_scan(thrust::device,&P_aux_offd_i[0],&P_aux_offd_i[n_fine+1],&P_aux_offd_i[0]);
      P_offd_size = P_aux_offd_i[n_fine];

      P_diag_j_new    = hypre_CTAlloc(HYPRE_Int,  P_diag_size, HYPRE_MEMORY_SHARED);
      P_diag_data_new = hypre_CTAlloc(HYPRE_Real, P_diag_size, HYPRE_MEMORY_SHARED);
      P_offd_j_new    = hypre_CTAlloc(HYPRE_Int,  P_offd_size, HYPRE_MEMORY_SHARED);
      P_offd_data_new = hypre_CTAlloc(HYPRE_Real, P_offd_size, HYPRE_MEMORY_SHARED);

      hypre_BoomerAMGInterpTruncationDevice_dev4<<<grid,block>>>( n_fine, P_diag_i, P_diag_j, P_diag_data, P_aux_diag_i,
							 P_diag_j_new, P_diag_data_new,
							 P_offd_i, P_offd_j, P_offd_data, P_aux_offd_i,
							 P_offd_j_new, P_offd_data_new );
      cudaDeviceSynchronize();
      //      P_diag_i[n_fine] = P_diag_size ;
      hypre_TFree(P_diag_i, HYPRE_MEMORY_SHARED);
      hypre_TFree(P_diag_j, HYPRE_MEMORY_SHARED);
      hypre_TFree(P_diag_data, HYPRE_MEMORY_SHARED);
      hypre_CSRMatrixI(P_diag) = P_aux_diag_i;
      hypre_CSRMatrixJ(P_diag) = P_diag_j_new;
      hypre_CSRMatrixData(P_diag) = P_diag_data_new;
      hypre_CSRMatrixNumNonzeros(P_diag) = P_diag_size;

      //      P_offd_i[n_fine] = P_offd_size ;
      hypre_TFree(P_offd_i, HYPRE_MEMORY_SHARED);
      hypre_TFree(P_offd_j, HYPRE_MEMORY_SHARED);
      hypre_TFree(P_offd_data, HYPRE_MEMORY_SHARED);
      hypre_CSRMatrixI(P_offd) = P_aux_offd_i;
      hypre_CSRMatrixJ(P_offd) = P_offd_j_new;
      hypre_CSRMatrixData(P_offd) = P_offd_data_new;
      hypre_CSRMatrixNumNonzeros(P_offd) = P_offd_size;
   }
   else if( P_aux_diag_i != NULL )
   {
      hypre_TFree(P_aux_diag_i, HYPRE_MEMORY_SHARED);
      hypre_TFree(P_aux_offd_i, HYPRE_MEMORY_SHARED);
   }
   return 0;
}


/* -----------------------------------------------------------------------*/
__global__ void hypre_BoomerAMGInterpTruncationDevice_dev1( HYPRE_Int num_rows_P,
							    HYPRE_Int* P_diag_i,
							    HYPRE_Int* P_diag_j,
							    HYPRE_Real* P_diag_data,
							    HYPRE_Int* P_offd_i,
							    HYPRE_Int* P_offd_j,
							    HYPRE_Real* P_offd_data,
							    HYPRE_Int* P_aux_diag_i,
							    HYPRE_Int* P_aux_offd_i,
							    HYPRE_Real trunc_factor )
   /*
    Perform truncation by eleminating all elements from row i whose absolute value is
    smaller than trunc_factor*max_k|P_{i,k}|.
    The matrix is rescaled after truncation to conserve its row sums.

    Input: num_rows_P - Number of rows of matrix in this MPI-task.
           P_diag_i, P_diag_j, P_diag_data - CSR representation of block diagonal part of matrix.
           P_offd_i, P_offd_j, P_offd_data - CSR representation of off-block diagonal part of matrix.
           trunc_factor - Factor in truncation threshold.

    Output:  P_aux_diag_i - P_aux_diag_i[i] holds the number of non-truncated elements on row i of P_diag.
	     P_aux_offd_i - P_aux_offd_i[i] holds the number of non-truncated elements on row i of P_offd.
	     P_diag_j, P_diag_data, P_offd_j, P_offd_data - For rows where truncation occurs, elements are
                 reordered to have the non-truncated elements first on each row, and the data arrays are rescaled.

   */
{
   HYPRE_Int myid= threadIdx.x + blockIdx.x * blockDim.x, i, ind, indp,nel_diag, nel_offd;
   HYPRE_Real max_coef, row_sum, row_sum_trunc;
   const HYPRE_Int nthreads = gridDim.x * blockDim.x;

   for( i = myid ; i < num_rows_P ; i += nthreads )
   {
  /* 1. Compute maximum absolute value element in row */
      max_coef = 0;
      for (ind = P_diag_i[i]; ind < P_diag_i[i+1]; ind++)
	 max_coef = (max_coef < fabs(P_diag_data[ind])) ?
	    fabs(P_diag_data[ind]) : max_coef;
      for (ind = P_offd_i[i]; ind < P_offd_i[i+1]; ind++)
	 max_coef = (max_coef < fabs(P_offd_data[ind])) ?
	    fabs(P_offd_data[ind]) : max_coef;
      max_coef *= trunc_factor;

  /* 2. Eliminate small elements and compress row */
      nel_diag = 0;
      row_sum = 0;
      row_sum_trunc = 0;
      indp = P_diag_i[i];
      for (ind = P_diag_i[i]; ind < P_diag_i[i+1]; ind++)
      {
	 row_sum += P_diag_data[ind];
	 if( fabs(P_diag_data[ind]) >= max_coef)
	 {
	    row_sum_trunc += P_diag_data[ind];
	    P_diag_data[indp+nel_diag] = P_diag_data[ind];
	    P_diag_j[indp+nel_diag++]  = P_diag_j[ind];
	 }
      }
      nel_offd = 0;
      indp=P_offd_i[i];
      for (ind = P_offd_i[i]; ind < P_offd_i[i+1]; ind++)
      {
	 row_sum += P_offd_data[ind];
	 if( fabs(P_offd_data[ind]) >= max_coef)
	 {
	    row_sum_trunc += P_offd_data[ind];
	    P_offd_data[indp+nel_offd] = P_offd_data[ind];
	    P_offd_j[indp+nel_offd++]  = P_offd_j[ind];
	 }
      }

  /* 3. Rescale row to conserve row sum */
      if( row_sum_trunc != 0 )
      {
	 if( row_sum_trunc != row_sum )
	 {
	    row_sum_trunc = row_sum/row_sum_trunc;
	    for (ind = P_diag_i[i]; ind < P_diag_i[i]+nel_diag; ind++)
	       P_diag_data[ind] *= row_sum_trunc;
	    for (ind = P_offd_i[i]; ind < P_offd_i[i]+nel_offd; ind++)
	       P_offd_data[ind] *= row_sum_trunc;
	 }
      }
  /* 4. Remember number of elements of compressed matrix */
      P_aux_diag_i[i] = nel_diag;
      P_aux_offd_i[i] = nel_offd;
   }
}

/*-----------------------------------------------------------------------*/
__global__ void hypre_BoomerAMGInterpTruncationDevice_dev2( HYPRE_Int  num_rows_P,
							    HYPRE_Int* P_diag_i,
							    HYPRE_Int* P_offd_i,
							    HYPRE_Int* P_aux_diag_i,
							    HYPRE_Int* P_aux_offd_i )
/*
   Construct P_aux_diag_i and P_aux_offd_i from a non-truncated matrix.

   Input: num_rows_P - Number of rows of matrix in this MPI-task.
          P_diag_i - CSR vector I of P_diag.
          P_offd_i - CSR vector I of P_offd.

   Output: P_aux_diag_i - P_aux_diag_i[i] holds the number of elements on row i in P_diag.
	   P_aux_offd_i - P_aux_offd_i[i] holds the number of elements on row i in P_offd.
 */
{
   HYPRE_Int myid= threadIdx.x + blockIdx.x * blockDim.x, i;
   const HYPRE_Int nthreads = gridDim.x * blockDim.x;
   for( i = myid ; i < num_rows_P ; i += nthreads )
      P_aux_diag_i[i] = P_diag_i[i+1]-P_diag_i[i];
   for( i = myid ; i < num_rows_P ; i += nthreads )
      P_aux_offd_i[i] = P_offd_i[i+1]-P_offd_i[i];
}

/*-----------------------------------------------------------------------*/
__global__ void hypre_BoomerAMGInterpTruncationDevice_dev3( HYPRE_Int   num_rows_P,
							    HYPRE_Int*  P_diag_i,
							    HYPRE_Int*  P_diag_j,
							    HYPRE_Real* P_diag_data,
							    HYPRE_Int*  P_offd_i,
							    HYPRE_Int*  P_offd_j,
							    HYPRE_Real* P_offd_data,
							    HYPRE_Int*  P_aux_diag_i,
							    HYPRE_Int*  P_aux_offd_i,
							    HYPRE_Int   max_elements )
   /*
    Perform truncation by retaining the max_elements largest (absolute value) elements of each row.
    The matrix is rescaled after truncation to conserve its row sums.

    Input: num_rows_P - Number of rows of matrix in this MPI-task.
           P_diag_i, P_diag_j, P_diag_data - CSR representation of block diagonal part of matrix
           P_offd_i, P_offd_j, P_offd_data - CSR representation of off-block diagonal part of matrix
	   P_aux_diag_i - P_aux_diag_i[i] holds the number of non-truncated elements on row i in P_diag.
	   P_aux_offd_i - P_aux_offd_i[i] holds the number of non-truncated elements on row i in P_offd.

    Output: P_aux_diag_i, P_aux_offd_i - Updated with the new number of elements per row, after truncation.
            P_diag_j, P_diag_data, P_offd_j, P_offd_data - Reordered so that the first P_aux_diag_i[i] and
                                  the first P_aux_offd_i[i] elements on each row form the truncated matrix.
*/
{
   HYPRE_Int myid= threadIdx.x + blockIdx.x * blockDim.x, i, nel, ind, indo;
   HYPRE_Real row_sum, row_sum_trunc;
   const HYPRE_Int nthreads = gridDim.x * blockDim.x;

   for( i = myid ; i < num_rows_P ; i += nthreads )
   {
   /* count number of elements in row */
      nel = P_aux_diag_i[i]+P_aux_offd_i[i];

  /* 0. Do we need to do anything ? */
      if( nel > max_elements )
      {
  /* 1. Save row sum before truncation, for rescaling below */
         row_sum = 0;
	 for (ind = P_diag_i[i]; ind < P_diag_i[i]+P_aux_diag_i[i]; ind++)
	    row_sum += P_diag_data[ind];
	 for (ind = P_offd_i[i]; ind < P_offd_i[i]+P_aux_offd_i[i]; ind++)
	    row_sum += P_offd_data[ind];

	 /* Sort in place, avoid allocation of extra array */
	 /* The routine hypre_qsort2abs(v,w,i0,i1) sorts (v,w) in decreasing order w.r.t w */
	 hypre_qsort2abs_dev(&P_diag_j[i], &P_diag_data[i], 0, P_aux_diag_i[i]-1 );
	 hypre_qsort2abs_dev(&P_offd_j[i], &P_offd_data[i], 0, P_aux_offd_i[i]-1 );

  /* 2. Retain the max_elements largest elements, only index of last element
        needs to be computed, since data is now sorted                        */
	 nel = 0;
	 ind =P_diag_i[i];
	 indo=P_offd_i[i];

  /* 2a. Also, keep track of row sum of truncated matrix, for rescaling below */
	 row_sum_trunc = 0;
	 while( nel < max_elements )
	 {
	    if( ind  < P_diag_i[i]+P_aux_diag_i[i] && indo < P_offd_i[i]+P_aux_offd_i[i] )
	    {
	       if( fabs(P_diag_data[ind])>fabs(P_offd_data[indo]) )
	       {
		  row_sum_trunc += P_diag_data[ind];
		  ind++;
	       }
	       else
	       {
		  row_sum_trunc += P_offd_data[indo];
		  indo++;
	       }
	    }
	    else if( ind < P_diag_i[i]+P_aux_diag_i[i] )
	    {
	       row_sum_trunc += P_diag_data[ind];
	       ind++;
	    }
	    else
	    {
	       row_sum_trunc += P_offd_data[indo];
	       indo++;
	    }
	    nel++;
	 }
  /* 3. Remember new row sizes */
	 P_aux_diag_i[i] = ind-P_diag_i[i];
	 P_aux_offd_i[i] = indo-P_offd_i[i];

  /* 4. Rescale row to conserve row sum */
	 if( row_sum_trunc != 0 )
	 {
	    if( row_sum_trunc != row_sum )
	    {
	       row_sum_trunc = row_sum/row_sum_trunc;
	       for (ind = P_diag_i[i]; ind < P_diag_i[i]+P_aux_diag_i[i]; ind++)
		  P_diag_data[ind] *= row_sum_trunc;
	       for (ind = P_offd_i[i]; ind < P_offd_i[i]+P_aux_offd_i[i]; ind++)
		  P_offd_data[ind] *= row_sum_trunc;
	    }
	 }
      }
   }
}


/*-----------------------------------------------------------------------*/
__global__ void hypre_BoomerAMGInterpTruncationDevice_dev4( HYPRE_Int   num_rows_P,
							    HYPRE_Int*  P_diag_i,
							    HYPRE_Int*  P_diag_j,
							    HYPRE_Real* P_diag_data,
							    HYPRE_Int*  P_diag_i_new,
							    HYPRE_Int*  P_diag_j_new,
							    HYPRE_Real* P_diag_data_new,
							    HYPRE_Int*  P_offd_i,
							    HYPRE_Int*  P_offd_j,
							    HYPRE_Real* P_offd_data,
							    HYPRE_Int*  P_offd_i_new,
							    HYPRE_Int*  P_offd_j_new,
							    HYPRE_Real* P_offd_data_new )
/*
   Copy truncated matrix to smaller storage. In the previous kernels, the number of elements per row
   has been reduced, but the matrix is still stored in the old CSR arrays.

   Input:  num_rows_P - Number of rows of matrix in this MPI-task.
           P_diag_i, P_diag_j, P_diag_data - CSR representation of block diagonal part of matrix
           P_offd_i, P_offd_j, P_offd_data - CSR representation of off-block diagonal part of matrix
           P_diag_i_new - P_diag has been truncated, this is the new CSR I-vector, pointing to beginnings of rows.
           P_offd_i_new - P_offd has been truncated, this is the new CSR I-vector, pointing to beginnings of rows.

   Output: P_diag_j_new, P_diag_data_new, P_offd_j_new, P_offd_data_new - These are the resized CSR arrays of the truncated matrix.
 */
{
   HYPRE_Int myid= threadIdx.x + blockIdx.x * blockDim.x, i, ind, indp, indo;
   const HYPRE_Int nthreads = gridDim.x * blockDim.x;

   for( i = myid ; i < num_rows_P ; i += nthreads )
   {
      //      indp = P_diag_i[i];
      indp = P_diag_i[i]-P_diag_i_new[i];
      for( ind = P_diag_i_new[i] ; ind < P_diag_i_new[i+1]; ind++ )
      {
	 //	 P_diag_j_new[ind]    = P_diag_j[indp];
	 //	 P_diag_data_new[ind] = P_diag_data[indp++];
	 P_diag_j_new[ind]    = P_diag_j[indp+ind];
	 P_diag_data_new[ind] = P_diag_data[indp+ind];
      }
      //      indo = P_offd_i[i];
      indo = P_offd_i[i]-P_offd_i_new[i];
      for( ind = P_offd_i_new[i] ; ind < P_offd_i_new[i+1]; ind++ )
      {
	 //	 P_offd_j_new[ind]    = P_offd_j[indo];
	 //	 P_offd_data_new[ind] = P_offd_data[indo++];
	 P_offd_j_new[ind]    = P_offd_j[indo+ind];
	 P_offd_data_new[ind] = P_offd_data[indo+ind];
      }
   }
}

/*-----------------------------------------------------------------------*/
__device__ void hypre_swap2_dev(HYPRE_Int  *v,
			    HYPRE_Real *w,
			    HYPRE_Int  i,
			    HYPRE_Int  j )
{
   HYPRE_Int temp;
   HYPRE_Real temp2;

   temp = v[i];
   v[i] = v[j];
   v[j] = temp;
   temp2 = w[i];
   w[i] = w[j];
   w[j] = temp2;
}

/*-----------------------------------------------------------------------*/
/* sort both v and w, in place, but based only on entries in w */
__device__ void hypre_qsort2abs_dev( HYPRE_Int *v,
				     HYPRE_Real *w,
				     HYPRE_Int  left,
				     HYPRE_Int  right )
{
   HYPRE_Int i, last;
   if (left >= right)
      return;
   hypre_swap2_dev( v, w, left, (left+right)/2);
   last = left;
   for (i = left+1; i <= right; i++)
      if (fabs(w[i]) > fabs(w[left]))
      {
         hypre_swap2_dev(v, w, ++last, i);
      }
   hypre_swap2_dev(v, w, left, last);
   hypre_qsort2abs_dev(v, w, left, last-1);
   hypre_qsort2abs_dev(v, w, last+1, right);
}

#endif

