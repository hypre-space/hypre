/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

#if defined(HYPRE_USING_CUDA)

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

/*---------------------------------------------------------------------------
 * hypre_BoomerAMGBuildDirInterp
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGBuildDirInterpDevice( hypre_ParCSRMatrix   *A,
                                     HYPRE_Int            *CF_marker,
                                     hypre_ParCSRMatrix   *S,
                                     HYPRE_BigInt         *num_cpts_global,
                                     HYPRE_Int             num_functions,
                                     HYPRE_Int            *dof_func,
                                     HYPRE_Int             debug_flag,
                                     HYPRE_Real            trunc_factor,
                                     HYPRE_Int             max_elmts,
                                     HYPRE_Int            *col_offd_S_to_A,
                                     hypre_ParCSRMatrix  **P_ptr)
{

   MPI_Comm          comm = hypre_ParCSRMatrixComm(A);
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
   //   HYPRE_BigInt   *col_map_offd_A = hypre_ParCSRMatrixColMapOffd(A);

   hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int       *S_diag_i = hypre_CSRMatrixI(S_diag);
   HYPRE_Int       *S_diag_j = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix *S_offd = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int       *S_offd_i = hypre_CSRMatrixI(S_offd);
   HYPRE_Int       *S_offd_j = hypre_CSRMatrixJ(S_offd);

   hypre_ParCSRMatrix *P;
   HYPRE_BigInt       *col_map_offd_P;
   HYPRE_Int          *tmp_map_offd = NULL;

   HYPRE_Int          *CF_marker_dev = NULL;
   /*   HYPRE_Int          *CF_marker_host = NULL;*/
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

   //   HYPRE_Int        prproc=2;
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

   ///* 0. Assume CF_marker has been allocated in device memory */
   //   CF_marker_host = hypre_CTAlloc(HYPRE_Int,  n_fine, HYPRE_MEMORY_HOST);
   //   hypre_TMemcpy( CF_marker_host, CF_marker, HYPRE_Int, n_fine, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE );
/* 0. Assume CF_marker has been allocated in host memory */
//   CF_marker_host = CF_marker;
   CF_marker_dev = hypre_CTAlloc(HYPRE_Int,  n_fine, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy( CF_marker_dev, CF_marker, HYPRE_Int, n_fine, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST );


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
            // = CF_marker_host[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
            = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
   }

   comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data, CF_marker_offd);
   hypre_ParCSRCommHandleDestroy(comm_handle);
   //   hypre_TFree(CF_marker_host, HYPRE_MEMORY_HOST);

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

   HYPRE_CUDA_LAUNCH( hypre_BoomerAMGBuildDirInterp_dev1, grid, block,
                      n_fine, S_diag_i, S_diag_j, S_offd_i, S_offd_j,
                      A_offd_i, A_offd_j,
                      CF_marker_dev, CF_marker_offd, num_functions,
                      dof_func, dof_func_offd, P_diag_i, P_offd_i,
                      col_offd_S_to_A, fine_to_coarse );

 /* The scans will transform P_diag_i and P_offd_i to the CSR I-vectors */
   HYPRE_THRUST_CALL(exclusive_scan, &P_diag_i[0], &P_diag_i[n_fine+1], &P_diag_i[0] );
   HYPRE_THRUST_CALL(exclusive_scan, &P_offd_i[0], &P_offd_i[n_fine+1], &P_offd_i[0] );
/* The scan will make fine_to_coarse[i] for i a coarse point hold a coarse point index in the range from 0 to n_coarse-1 */
   HYPRE_THRUST_CALL(exclusive_scan, &fine_to_coarse[0], &fine_to_coarse[n_fine], &fine_to_coarse[0] );

/* 4. Compute the CSR arrays P_diag_j, P_diag_data, P_offd_j, and P_offd_data */
/*    P_diag_i and P_offd_i are now known, first allocate the remaining CSR arrays of P */

   P_diag_size = P_diag_i[n_fine];
   P_diag_j    = hypre_CTAlloc(HYPRE_Int,  P_diag_size, HYPRE_MEMORY_SHARED);
   P_diag_data = hypre_CTAlloc(HYPRE_Real, P_diag_size, HYPRE_MEMORY_SHARED);

   P_offd_size = P_offd_i[n_fine];
   P_offd_j    = hypre_CTAlloc(HYPRE_Int,  P_offd_size, HYPRE_MEMORY_SHARED);
   P_offd_data = hypre_CTAlloc(HYPRE_Real, P_offd_size, HYPRE_MEMORY_SHARED);

   HYPRE_CUDA_LAUNCH( hypre_BoomerAMGBuildDirInterp_dev2, grid, block,
                      n_fine, A_diag_i, A_diag_j, A_diag_data,
                      A_offd_i, A_offd_j, A_offd_data,
                      S_diag_i, S_diag_j, S_offd_i, S_offd_j,
                      CF_marker_dev, CF_marker_offd,
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

      //      P_diag_data = hypre_CSRMatrixData(P_diag);
      //      P_diag_i = hypre_CSRMatrixI(P_diag);
      //      P_diag_j = hypre_CSRMatrixJ(P_diag);


      P_offd_data = hypre_CSRMatrixData(P_offd);
      P_offd_i = hypre_CSRMatrixI(P_offd);
      P_offd_j = hypre_CSRMatrixJ(P_offd);
      P_offd_size=P_offd_i[n_fine];
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
      HYPRE_Int *P_marker = hypre_CTAlloc(HYPRE_Int, num_cols_A_offd+1, HYPRE_MEMORY_DEVICE);
      HYPRE_CUDA_LAUNCH( hypre_BoomerAMGBuildDirInterp_dev5, grid, block,
                         P_offd_size, P_offd_j, P_marker );

      /* Secondly, the sum over P_marker gives the number of different columns in P's offd part */
      num_cols_P_offd = HYPRE_THRUST_CALL(reduce, &P_marker[0], &P_marker[num_cols_A_offd]);

      /* Because P's columns correspond to P_marker[i]=1 (and =0 otherwise), the scan below will return  */
      /* an enumeration of P's columns 0,1,... at the corresponding locations in P_marker. */
      /* P_marker[num_cols_A_offd] will contain num_cols_P_offd, so sum reduction above could  */
      /* have been replaced by reading the last element of P_marker. */
      HYPRE_THRUST_CALL(exclusive_scan, &P_marker[0], &P_marker[num_cols_A_offd+1], &P_marker[0] );
      /* Example: P_marker becomes [0,0,1,1,1,2] so that P_marker[1]=0, P_marker[4]=1  */

      /* Do the re-enumeration, P_offd_j are mapped, using P_marker as map  */
      HYPRE_CUDA_LAUNCH( hypre_BoomerAMGBuildDirInterp_dev3, grid, block,
                         P_offd_size, P_offd_j, P_marker );

      /* Create and define array tmp_map_offd. This array is the inverse of the P_marker mapping, */
      /* Example: num_cols_P_offd=2, tmp_map_offd[0] = 1, tmp_map_offd[1]=4  */
      tmp_map_offd   = hypre_CTAlloc(HYPRE_Int, num_cols_P_offd, HYPRE_MEMORY_SHARED);
      grid.x = num_cols_A_offd/block.x;
      if( num_cols_A_offd % block.x != 0 )
         grid.x++;
      if( grid.x > limit )
         grid.x = limit;

      HYPRE_CUDA_LAUNCH( hypre_BoomerAMGBuildDirInterp_dev4, grid, block,
                         num_cols_A_offd, P_marker, tmp_map_offd );

      if (num_cols_P_offd)
      {
         col_map_offd_P = hypre_CTAlloc(HYPRE_BigInt, num_cols_P_offd, HYPRE_MEMORY_HOST);
         // col_map_offd_P = hypre_CTAlloc(HYPRE_BigInt, num_cols_P_offd, HYPRE_MEMORY_SHARED);
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

   hypre_TFree(CF_marker_dev, HYPRE_MEMORY_DEVICE);
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
       HYPRE_Int* A_offd_i, HYPRE_Int* A_offd_j,
       HYPRE_Int* CF_marker, HYPRE_Int* CF_marker_offd,
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
               //       if (A_diag_data[ind] > 0)
               //          sum_N_pos += A_diag_data[ind];
               //       else
               //          sum_N_neg += A_diag_data[ind];
            }
            if( inds < S_diag_i[i+1] && i1==S_diag_j[inds] )
            {
               /* Element is in both A and S */
               if (CF_marker[i1] > 0 && ( num_functions==1 || dof_func[i1]==dof_func[i]) )
               {
                  //  P_diag_data[indp] = A_diag_data[ind];
                  P_diag_data[indp] = aval;
                  P_diag_j[indp++]  = fine_to_coarse[i1];
                  //  if( A_diag_data[ind] > 0 )
                  //     sum_P_pos += A_diag_data[ind];
                  //  else
                  //     sum_P_neg += A_diag_data[ind];
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

#endif

