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

#if defined(HYPRE_USING_CUDA)

 __global__ void hypre_BoomerAMGCreateS_dev1b( HYPRE_Int nr_of_rows, HYPRE_Real max_row_sum, HYPRE_Real strength_threshold,
					HYPRE_Real* A_diag_data, HYPRE_Int* A_diag_i, HYPRE_Int* A_diag_j,
					HYPRE_Real* A_offd_data, HYPRE_Int* A_offd_i, HYPRE_Int* A_offd_j,
                                        HYPRE_Int* S_temp_diag_j, HYPRE_Int* S_temp_offd_j,
					       HYPRE_Int* jS_diag, HYPRE_Int* jS_offd );

__global__ void hypre_BoomerAMGCreateS_dev1( HYPRE_Int nr_of_rows,
					     HYPRE_Real max_row_sum, HYPRE_Real strength_threshold,
					HYPRE_Real* A_diag_data, HYPRE_Int* A_diag_i, HYPRE_Int* A_diag_j,
					HYPRE_Real* A_offd_data, HYPRE_Int* A_offd_i, HYPRE_Int* A_offd_j,
                                        HYPRE_Int* S_temp_diag_j, HYPRE_Int* S_temp_offd_j,
  //					HYPRE_Int num_functions, HYPRE_Int* dof_func, HYPRE_Int* dof_func_offd,
					HYPRE_Int* jS_diag, HYPRE_Int* jS_offd );


 __global__ void hypre_BoomerAMGCreateS_dev1_mf( HYPRE_Int nr_of_rows, HYPRE_Real max_row_sum, HYPRE_Real strength_threshold,
					HYPRE_Real* A_diag_data, HYPRE_Int* A_diag_i, HYPRE_Int* A_diag_j,
					HYPRE_Real* A_offd_data, HYPRE_Int* A_offd_i, HYPRE_Int* A_offd_j,
                                        HYPRE_Int* S_temp_diag_j, HYPRE_Int* S_temp_offd_j,
					HYPRE_Int num_functions, HYPRE_Int* dof_func, HYPRE_Int* dof_func_offd,
						 HYPRE_Int* jS_diag, HYPRE_Int* jS_offd );

__global__ void hypre_BoomerAMGCreateS_dev2( HYPRE_Int nr_of_rows, HYPRE_Int* A_diag_i, HYPRE_Int* A_offd_i,
					HYPRE_Int* S_diag_i, HYPRE_Int* S_diag_j, HYPRE_Int* S_temp_diag_j,
					HYPRE_Int* S_offd_i, HYPRE_Int* S_offd_j, HYPRE_Int* S_temp_offd_j );

HYPRE_Int
hypre_BoomerAMGCreateSDevice(hypre_ParCSRMatrix    *A,
			     HYPRE_Real             strength_threshold,
			     HYPRE_Real             max_row_sum,
			     HYPRE_Int              num_functions,
			     HYPRE_Int             *dof_func,
			     hypre_ParCSRMatrix   **S_ptr)
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_CREATES] -= hypre_MPI_Wtime();
#endif
   //PUSH_RANGE("CreateS_dev",0);
   MPI_Comm 	       comm            = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg     *comm_pkg   = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;
   hypre_CSRMatrix    *A_diag          = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int                *A_diag_i        = hypre_CSRMatrixI(A_diag);
   HYPRE_Real         *A_diag_data     = hypre_CSRMatrixData(A_diag);


   hypre_CSRMatrix    *A_offd          = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int                *A_offd_i        = hypre_CSRMatrixI(A_offd);
   HYPRE_Real         *A_offd_data = NULL;
   HYPRE_Int                *A_diag_j        = hypre_CSRMatrixJ(A_diag);
   HYPRE_Int                *A_offd_j        = hypre_CSRMatrixJ(A_offd);

   HYPRE_Int 		      *row_starts      = hypre_ParCSRMatrixRowStarts(A);
   HYPRE_Int                 num_variables   = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int                 global_num_vars = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_Int 		       num_nonzeros_diag;
   HYPRE_Int 		       num_nonzeros_offd = 0;
   HYPRE_Int 		       num_cols_offd = 0;

   hypre_ParCSRMatrix *S;
   hypre_CSRMatrix    *S_diag;
   HYPRE_Int                *S_diag_i;
   HYPRE_Int                *S_diag_j;
   /* HYPRE_Real         *S_diag_data; */
   hypre_CSRMatrix    *S_offd;
   HYPRE_Int                *S_offd_i = NULL;
   HYPRE_Int                *S_offd_j = NULL;
   /* HYPRE_Real         *S_offd_data; */

   //   HYPRE_Real          diag, row_scale, row_sum;
   HYPRE_Int                 i;

   HYPRE_Int                 ierr = 0;

   HYPRE_Int                 *dof_func_offd=NULL;
   HYPRE_Int                 *dof_func_offd_dev=NULL;
   HYPRE_Int                 *dof_func_dev=NULL;
   HYPRE_Int			num_sends;
   HYPRE_Int		       *int_buf_data;
   HYPRE_Int			index, start, j;

   //   HYPRE_Int *prefix_sum_workspace;

   /*--------------------------------------------------------------
    * Compute a  ParCSR strength matrix, S.
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

   num_nonzeros_diag = A_diag_i[num_variables];
   num_cols_offd = hypre_CSRMatrixNumCols(A_offd);

   A_offd_i = hypre_CSRMatrixI(A_offd);
   num_nonzeros_offd = A_offd_i[num_variables];

   S = hypre_ParCSRMatrixCreate(comm, global_num_vars, global_num_vars,
			row_starts, row_starts,
			num_cols_offd, num_nonzeros_diag, num_nonzeros_offd);
/* row_starts is owned by A, col_starts = row_starts */
   hypre_ParCSRMatrixSetRowStartsOwner(S,0);
   S_diag = hypre_ParCSRMatrixDiag(S);
   hypre_CSRMatrixI(S_diag) = hypre_CTAlloc(HYPRE_Int,  num_variables+1, HYPRE_MEMORY_SHARED);
   hypre_CSRMatrixJ(S_diag) = hypre_CTAlloc(HYPRE_Int,  num_nonzeros_diag, HYPRE_MEMORY_SHARED);
   S_offd = hypre_ParCSRMatrixOffd(S);
   hypre_CSRMatrixI(S_offd) = hypre_CTAlloc(HYPRE_Int,  num_variables+1, HYPRE_MEMORY_SHARED);

   S_diag_i = hypre_CSRMatrixI(S_diag);
   HYPRE_Int *S_temp_diag_j = hypre_CSRMatrixJ(S_diag);
   S_offd_i = hypre_CSRMatrixI(S_offd);

   S_diag_j = hypre_TAlloc(HYPRE_Int,  num_nonzeros_diag, HYPRE_MEMORY_SHARED);
   HYPRE_Int *S_temp_offd_j = NULL;

   if (num_cols_offd)
   {
        A_offd_data = hypre_CSRMatrixData(A_offd);
        hypre_CSRMatrixJ(S_offd) = hypre_CTAlloc(HYPRE_Int,  num_nonzeros_offd, HYPRE_MEMORY_SHARED);
        S_temp_offd_j = hypre_CSRMatrixJ(S_offd);
        HYPRE_Int *col_map_offd_S = hypre_TAlloc(HYPRE_Int,  num_cols_offd, HYPRE_MEMORY_HOST);
        hypre_ParCSRMatrixColMapOffd(S) = col_map_offd_S;
        if (num_functions > 1)
	{
	   dof_func_offd = hypre_CTAlloc(HYPRE_Int,  num_cols_offd, HYPRE_MEMORY_HOST);
	   dof_func_offd_dev = hypre_CTAlloc(HYPRE_Int,  num_cols_offd, HYPRE_MEMORY_DEVICE);
	}
        S_offd_j = hypre_TAlloc(HYPRE_Int,  num_nonzeros_offd, HYPRE_MEMORY_SHARED);

        HYPRE_Int *col_map_offd_A = hypre_ParCSRMatrixColMapOffd(A);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
        for (i = 0; i < num_cols_offd; i++)
           col_map_offd_S[i] = col_map_offd_A[i];
   }


  /*-------------------------------------------------------------------
    * Get the dof_func data for the off-processor columns
    *-------------------------------------------------------------------*/


   if (!comm_pkg)
   {
	hypre_MatvecCommPkgCreate(A);
	comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   if (num_functions > 1 )
   {
      int_buf_data = hypre_CTAlloc(HYPRE_Int, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
						num_sends), HYPRE_MEMORY_HOST);
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

      hypre_TFree(int_buf_data, HYPRE_MEMORY_HOST);
      hypre_TMemcpy( dof_func_offd_dev, dof_func_offd, HYPRE_Int, num_cols_offd, HYPRE_MEMORY_DEVICE,
                     HYPRE_MEMORY_HOST );
      dof_func_dev = hypre_CTAlloc(HYPRE_Int,  num_variables, HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy( dof_func_dev, dof_func, HYPRE_Int, num_variables, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST );
   }

   /*HYPRE_Int prefix_sum_workspace[2*(hypre_NumThreads() + 1)];*/
   /*   prefix_sum_workspace = hypre_TAlloc(HYPRE_Int,  2*(hypre_NumThreads() + 1), HYPRE_MEMORY_HOST);*/

   /* give S same nonzero structure as A */

   dim3 grid, block(32,1,1);
   grid.x = num_variables/block.x;
   if( num_variables % block.x != 0 )
      grid.x++;
   //   grid.x = num_variables;
   grid.y = 1;
   grid.z = 1;
   //   size_t shmemsize = block.x*(2*sizeof(HYPRE_Real)+2*sizeof(HYPRE_Int));
   //   size_t totshmem = 4*sizeof(HYPRE_Int)+sizeof(HYPRE_Real)+shmemsize;
   if( num_functions == 1 )
   {
      hypre_BoomerAMGCreateS_dev1<<<grid,block>>>( num_variables, max_row_sum, strength_threshold,
            						   A_diag_data, A_diag_i, A_diag_j,
            						   A_offd_data, A_offd_i, A_offd_j,
            						   S_temp_diag_j, S_temp_offd_j,
            						   S_diag_i, S_offd_i );
      //       hypre_BoomerAMGCreateS_dev1b<<<grid,block,shmemsize>>>( num_variables, max_row_sum, strength_threshold,
      // 						   A_diag_data, A_diag_i, A_diag_j,
      // 						   A_offd_data, A_offd_i, A_offd_j,
      // 						   S_temp_diag_j, S_temp_offd_j,
      //						   S_diag_i, S_offd_i );
   }
   else
      hypre_BoomerAMGCreateS_dev1_mf<<<grid,block>>>( num_variables, max_row_sum, strength_threshold,
						      A_diag_data, A_diag_i, A_diag_j,
						      A_offd_data, A_offd_i, A_offd_j,
						      S_temp_diag_j, S_temp_offd_j,
						      num_functions, dof_func_dev, dof_func_offd_dev,
						      S_diag_i, S_offd_i );

   cudaDeviceSynchronize();
   thrust::exclusive_scan( thrust::device, &S_diag_i[0], &S_diag_i[num_variables+1], &S_diag_i[0] );
   thrust::exclusive_scan( thrust::device, &S_offd_i[0], &S_offd_i[num_variables+1], &S_offd_i[0] );

   hypre_BoomerAMGCreateS_dev2<<<grid,block>>>( num_variables, A_diag_i, A_offd_i,
						S_diag_i, S_diag_j, S_temp_diag_j,
						S_offd_i, S_offd_j, S_temp_offd_j );
   cudaDeviceSynchronize();
   hypre_CSRMatrixNumNonzeros(S_diag) = S_diag_i[num_variables];
   hypre_CSRMatrixNumNonzeros(S_offd) = S_offd_i[num_variables];
   hypre_CSRMatrixJ(S_diag) = S_diag_j;
   hypre_CSRMatrixJ(S_offd) = S_offd_j;

   hypre_ParCSRMatrixCommPkg(S) = NULL;

   *S_ptr        = S;

   /*   hypre_TFree(prefix_sum_workspace, HYPRE_MEMORY_HOST);*/
   if( num_cols_offd >0 && num_functions > 1 )
   {
      hypre_TFree(dof_func_offd, HYPRE_MEMORY_HOST);
      hypre_TFree(dof_func_offd_dev, HYPRE_MEMORY_DEVICE);
   }
   if( num_functions > 1 )
      hypre_TFree(dof_func_dev, HYPRE_MEMORY_DEVICE);

   hypre_TFree(S_temp_diag_j, HYPRE_MEMORY_SHARED);
   if( num_cols_offd > 0 )
      hypre_TFree(S_temp_offd_j, HYPRE_MEMORY_SHARED);


#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_CREATES] += hypre_MPI_Wtime();
#endif
   //POP_RANGE
   return (ierr);
}

/*-----------------------------------------------------------------------*/
 __global__ void hypre_BoomerAMGCreateS_dev1( HYPRE_Int nr_of_rows, HYPRE_Real max_row_sum, HYPRE_Real strength_threshold,
					HYPRE_Real* A_diag_data, HYPRE_Int* A_diag_i, HYPRE_Int* A_diag_j,
					HYPRE_Real* A_offd_data, HYPRE_Int* A_offd_i, HYPRE_Int* A_offd_j,
                                        HYPRE_Int* S_temp_diag_j, HYPRE_Int* S_temp_offd_j,
  //					HYPRE_Int num_functions, HYPRE_Int* dof_func, HYPRE_Int* dof_func_offd,
					HYPRE_Int* jS_diag, HYPRE_Int* jS_offd )
{
   /*-----------------------------------------------------------------------*/
   /*
      Input: nr_of_rows - Number of rows in matrix (local in processor)
             A_diag_data, A_diag_i, A_diag_j - CSR representation of A_diag
             A_offd_data, A_offd_i, A_offd_j - CSR representation of A_offd
	     num_function  - Number of degrees of freedom per grid point
	     dof_func      - vector over nonzero elements of A_diag, indicating the degree of freedom
	     dof_func_offd - vector over nonzero elements of A_offd, indicating the degree of freedom

      Output: S_temp_diag_j - S_diag_j vector before compression, i.e.,elements that are -1 should be removed
              S_temp_offd_j - S_offd_j vector before compression, i.e.,elements that are -1 should be removed
	      jS_diag       - S_diag_i vector for compressed S_diag
	      jS_offd       - S_offd_i vector for compressed S_offd
    */
   /*-----------------------------------------------------------------------*/

   HYPRE_Real row_scale, row_sum, diag;
   //   HYPRE_Int i, myid = threadIdx.x + blockIdx.x * blockDim.x, jA, jSd, jSo, Adi, Adip, Aoi, Aoip, cond, notallweak;
   HYPRE_Int myid= threadIdx.x + blockIdx.x * blockDim.x, jA, jSd, jSo, Adi, Adip, Aoi, Aoip, cond, notallweak;
   HYPRE_Int i, sdiag;
   const HYPRE_Int nthreads = gridDim.x * blockDim.x;

   for( i = myid ; i < nr_of_rows ; i += nthreads )
      //   if( i < nr_of_rows )
   {
      Adi = A_diag_i[i];
      Adip= A_diag_i[i+1];

 /* compute scaling factor and row sum */
      row_scale = 0.0;
      diag = A_diag_data[Adi];
      row_sum = diag;
      sdiag = diag>0 ? 1:-1;
      for (jA = Adi+1; jA < Adip; jA++)
      {
	 row_scale = hypre_min(row_scale, sdiag*A_diag_data[jA]);
	 row_sum += A_diag_data[jA];
      }
      Aoi = A_offd_i[i];
      Aoip= A_offd_i[i+1];
      for (jA = Aoi; jA < Aoip; jA++)
      {
	 row_scale = hypre_min(row_scale, sdiag*A_offd_data[jA]);
	 row_sum += A_offd_data[jA];
      }
      row_scale *= sdiag;
      jSd=jSo=0;
      /* compute row entries of S */
      S_temp_diag_j[Adi] = -1;
      notallweak = !((fabs(row_sum) > sdiag*diag*max_row_sum) && (max_row_sum < 1.0));
      //      if ((fabs(row_sum) > fabs(diag)*max_row_sum) && (max_row_sum < 1.0))
      //      {
      //         /* make all dependencies weak */
      //         for (jA = Adi+1; jA < Adip; jA++)
      //            S_temp_diag_j[jA] = -1;
      //	 for (jA = Aoi; jA < Aoip; jA++)
      //	    S_temp_offd_j[jA] = -1;
      //      }
      //      else
      {
	 for (jA = Adi+1; jA < Adip; jA++)
	 {
            cond = notallweak && (sdiag*A_diag_data[jA] < sdiag * strength_threshold * row_scale);
	    //   	    cond = (sdiag*A_diag_data[jA] < sdiag * strength_threshold * row_scale);
		  //		  S_temp_diag_j[jA] = (!cond)*(-1) + cond*A_diag_j[jA];
	    S_temp_diag_j[jA]= cond*(1+A_diag_j[jA])-1;
	    jSd += cond;
	 }
	 for (jA = Aoi; jA < Aoip; jA++)
	 {
	    cond = notallweak && (sdiag*A_offd_data[jA] < sdiag * strength_threshold * row_scale);
	    //	    cond = (sdiag*A_offd_data[jA] < sdiag * strength_threshold * row_scale);
	    S_temp_offd_j[jA]= cond*(1+A_offd_j[jA])-1;
	    jSo += cond;
	 }
      } /* !((row_sum > max_row_sum) && (max_row_sum < 1.0)) */
      jS_diag[i] = jSd;
      jS_offd[i] = jSo;
   } /* for each variable */
}


/*-----------------------------------------------------------------------*/
__global__ void hypre_BoomerAMGCreateS_dev2( HYPRE_Int nr_of_rows, HYPRE_Int* A_diag_i, HYPRE_Int* A_offd_i,
					HYPRE_Int* S_diag_i, HYPRE_Int* S_diag_j, HYPRE_Int* S_temp_diag_j,
					HYPRE_Int* S_offd_i, HYPRE_Int* S_offd_j, HYPRE_Int* S_temp_offd_j )
{
   /* Create strength matrix */
   /*
      Input: nr_of_rows - Number of rows in matrix (local in processor)
             A_diag_i - vector A_diag_i in CSR representation of A_diag
             A_offd_i - vector A_offd_i in CSR representation of A_offd
	     S_diag_i - vector in CSR representation of S_diag, computed by hypre_BoomerAMGCreateS_dev1
	     S_offd_i - vector in CSR representation of S_offd, computed by hypre_BoomerAMGCreateS_dev1
	     S_temp_diag_j - S_diag_j vector before compression, computed by hypre_BoomerAMGCreateS_dev1
	     S_temp_offd_j - S_offd_j vector before compression, computed by hypre_BoomerAMGCreateS_dev1


      Output: S_diag_j - S_diag_j vector after compression
              S_offd_j - S_offd_j vector after compression
    */

   /*--------------------------------------------------------------
    * "Compress" the strength matrix.
    *
    * NOTE: S has *NO DIAGONAL ELEMENT* on any row.  Caveat Emptor!
    *
    * NOTE: This "compression" section of code may be removed, and
    * coarsening will still be done correctly.  However, the routine
    * that builds interpolation would have to be modified first.
    *----------------------------------------------------------------*/

   HYPRE_Int i, myid = threadIdx.x + blockIdx.x * blockDim.x, jA, jS;
   const HYPRE_Int nthreads = gridDim.x * blockDim.x;
   for( i = myid ; i < nr_of_rows ; i += nthreads )
   {
      jS = S_diag_i[i];
      for (jA = A_diag_i[i]; jA < A_diag_i[i+1]; jA++)
      {
         if (S_temp_diag_j[jA] > -1)
         {
            S_diag_j[jS] = S_temp_diag_j[jA];
            jS++;
         }
      }
      jS = S_offd_i[i];
      for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
      {
         if (S_temp_offd_j[jA] > -1)
         {
            S_offd_j[jS]    = S_temp_offd_j[jA];
            jS++;
         }
      }
   }
}

/*-----------------------------------------------------------------------*/
 __global__ void hypre_BoomerAMGCreateS_dev1b( HYPRE_Int nr_of_rows, HYPRE_Real max_row_sum, HYPRE_Real strength_threshold,
					HYPRE_Real* A_diag_data, HYPRE_Int* A_diag_i, HYPRE_Int* A_diag_j,
					HYPRE_Real* A_offd_data, HYPRE_Int* A_offd_i, HYPRE_Int* A_offd_j,
                                        HYPRE_Int* S_temp_diag_j, HYPRE_Int* S_temp_offd_j,
					HYPRE_Int* jS_diag, HYPRE_Int* jS_offd )
{
   /*-----------------------------------------------------------------------*/
   /* Experimental version of _dev1, this one did not show any gain in performance, do not use ...

      Input: nr_of_rows - Number of rows in matrix (local in processor)
             A_diag_data, A_diag_i, A_diag_j - CSR representation of A_diag
             A_offd_data, A_offd_i, A_offd_j - CSR representation of A_offd
	     num_function  - Number of degrees of freedom per grid point
	     dof_func      - vector over nonzero elements of A_diag, indicating the degree of freedom
	     dof_func_offd - vector over nonzero elements of A_offd, indicating the degree of freedom

      Output: S_temp_diag_j - S_diag_j vector before compression, i.e.,elements that are -1 should be removed
              S_temp_offd_j - S_offd_j vector before compression, i.e.,elements that are -1 should be removed
	      jS_diag       - S_diag_i vector for compressed S_diag
	      jS_offd       - S_offd_i vector for compressed S_offd
    */
   /*-----------------------------------------------------------------------*/

   HYPRE_Real row_scale, row_sum;
   HYPRE_Int i, cond, sdiag, nred, nredo, r, len, leno, ind;
   const HYPRE_Int myrow   = blockIdx.x;
   const HYPRE_Int myid    = threadIdx.x;
   //   const HYPRE_Int bzh     = blockDim.x/2;
   // Allocate shmem with block size*2 doubles
   __shared__ extern float shmem[];
   HYPRE_Real *shr, *shr2;
   HYPRE_Int *jSd, *jSo;
   // Shared memory variables
   //   __shared__ HYPRE_Int indbeg, indend, indbego, indendo;
   //   __shared__ HYPRE_Real diag;
   HYPRE_Int indbeg, indend, indbego, indendo;
   HYPRE_Real diag;


   shr = (HYPRE_Real*)shmem;
   shr2= (HYPRE_Real*)&shr[blockDim.x];
   jSd = (HYPRE_Int*)&shr2[blockDim.x];
   jSo =  (HYPRE_Int*)&jSd[blockDim.x];

   shr[myid]=shr2[myid]=0;
   for( i = myrow ; i < nr_of_rows ; i += gridDim.x )
      //   i = myrow;
      //   if( i < nr_of_rows )
   {
      //      if( myid==0)
      {
	 /* Declare these variables in shared memory */
	 indbeg  = A_diag_i[i];
	 indend  = A_diag_i[i+1];
	 indbego = A_offd_i[i];
	 indendo = A_offd_i[i+1];
	 diag    = A_diag_data[indbeg];
      }
      //      shr[myid]=shr2[myid]=0;
      //      __syncthreads();

      len = indend-indbeg;
      leno= indendo-indbego;
      sdiag = diag>0 ? 1:-1;
      nred = len/blockDim.x;
      nred += (len % blockDim.x != 0);
      nredo = leno/blockDim.x;
      nredo += (leno % blockDim.x != 0);

      ind =indbeg+myid;
      if( ind < indend )
      {
	 shr[myid]  = A_diag_data[ind];
	 shr2[myid] = myid == 0 ? 0 : sdiag*A_diag_data[ind];
      }
      for( r=1 ; r < nred ; r++ )
      {
	 ind += blockDim.x;
	 if( ind < indend )
	 {
	    shr[myid] += A_diag_data[ind];
	    shr2[myid] = hypre_min(shr2[myid],sdiag*A_diag_data[ind]);
	 }
      }

      /* update shr,shr2 with offd part of matrix */
      ind    = indbego+myid;
      for( r=0 ; r < nredo ; r++ )
      {
	 if( ind < indendo )
	 {
	    shr[myid]  += A_offd_data[ind];
	    shr2[myid] = hypre_min(shr2[myid],sdiag*A_offd_data[ind]);
	 }
	 ind += blockDim.x;
      }
      /* sum shr memory over threads in block */
      /* take minumum of shr2 memory over threads in block */
      __syncthreads();

      //      for( r=blockDim.x/2 ; r>0 ;r /= 2)
      //      for( r= blockDim.x/2 ; r>0 ; r >>= 1)
      for( r= blockDim.x/2 ; r>=32 ; r >>= 1)
      {
	 if( myid < r )
	 {
	    shr[myid] += shr[myid+r];
	    shr2[myid] = hypre_min(shr2[myid],shr2[myid+r]);
	 }
	 __syncthreads();
      }
      //      if( myid < 32 )
      if( myid < 16 )
      {

	 //      shr[myid] += shr[myid+32];
	 //      shr2[myid] = hypre_min(shr2[myid],shr2[myid+32]);
      shr[myid] += shr[myid+16];
      shr2[myid] = hypre_min(shr2[myid],shr2[myid+16]);
      shr[myid] += shr[myid+8];
      shr2[myid] = hypre_min(shr2[myid],shr2[myid+8]);
      shr[myid] += shr[myid+4];
      shr2[myid] = hypre_min(shr2[myid],shr2[myid+4]);
      shr[myid] += shr[myid+2];
      shr2[myid] = hypre_min(shr2[myid],shr2[myid+2]);
      shr[myid] += shr[myid+1];
      shr2[myid] = hypre_min(shr2[myid],shr2[myid+1]);
      }
      row_sum   = shr[0];
      row_scale = sdiag*shr2[0];

      //      __syncthreads();
      /*      jSd=jSo=0;*/
      /* compute row entries of S */
      /*      S_temp_diag_j[Adi] = -1;*/
      //      notallweak = !((fabs(row_sum) > sdiag*diag*max_row_sum) && (max_row_sum < 1.0));

      jSd[myid]=0;
      jSo[myid]=0;
      if ((fabs(row_sum) > fabs(diag)*max_row_sum) && (max_row_sum < 1.0))
      {
         /* make all dependencies weak */
	 ind    = indbeg+myid;
	 for( r=0 ; r < nred ; r++ )
	 {
	    if( ind < indend )
	       S_temp_diag_j[ind]  = -1;
	    ind += blockDim.x;
	 }
	 ind    = indbego+myid;
	 for( r=0 ; r < nredo ; r++ )
	 {
	    if( ind < indendo )
	       S_temp_offd_j[ind]  = -1;
	    ind += blockDim.x;
	 }
      }
      else
      {
	 ind    = indbeg+myid;
	 if( ind < indend )
	 {
	    cond = (myid !=0) && (sdiag*A_diag_data[ind] < sdiag * strength_threshold * row_scale);
	    S_temp_diag_j[ind]= cond*(1+A_diag_j[ind])-1;
	    jSd[myid] = cond;
	 }
	 for( r=1 ; r < nred ; r++ )
	 {
	    //	    ind += blockDim.x;
	    if( ind+r*blockDim.x < indend )
	    {
	       cond = sdiag*A_diag_data[ind+r*blockDim.x] < sdiag * strength_threshold * row_scale;
	       S_temp_diag_j[ind+r*blockDim.x]= cond*(1+A_diag_j[ind+r*blockDim.x])-1;
	       jSd[myid] += cond;
	    }
	 }
	 ind    = indbego+myid;
	 for( r=0 ; r < nredo ; r++ )
	 {
	    if( ind + r*blockDim.x < indendo )
	    {
	       cond = sdiag*A_offd_data[ind+r*blockDim.x] < sdiag * strength_threshold * row_scale;
	       S_temp_offd_j[ind+r*blockDim.x]= cond*(1+A_offd_j[ind+r*blockDim.x])-1;
	       jSo[myid] += cond;
	    }
	    //	    ind += blockDim.x;
	 }

      } /* !((row_sum > max_row_sum) && (max_row_sum < 1.0)) */
      shr[myid]=shr2[myid]=0; // pre compute
	 __syncthreads();
      for( r=blockDim.x/2 ; r>=32 ;r >>= 2)
	 //      for( r=blockDim.x/2 ; r>0 ;r /= 2)
      {
	 if( myid < r )
	 {
	    jSd[myid] += jSd[myid+r];
	    jSo[myid] += jSo[myid+r];
	 }
	 __syncthreads();
      }
      if( myid < 16 )
	 //      if( myid < 32 )
      {
	 //      jSd[myid] += jSd[myid+32];
	 //      jSo[myid] += jSo[myid+32];
      jSd[myid] += jSd[myid+16];
      jSo[myid] += jSo[myid+16];
      jSd[myid] += jSd[myid+8];
      jSo[myid] += jSo[myid+8];
      jSd[myid] += jSd[myid+4];
      jSo[myid] += jSo[myid+4];
      jSd[myid] += jSd[myid+2];
      jSo[myid] += jSo[myid+2];
      jSd[myid] += jSd[myid+1];
      jSo[myid] += jSo[myid+1];
      }

      if( myid==0 )
      {
	 jS_diag[i] = jSd[0];
	 jS_offd[i] = jSo[0];
      }

      __syncthreads();
      /*      jS_diag[i] = jSd; */
      /*      jS_offd[i] = jSo; */
   } /* for each variable */
}


/*-----------------------------------------------------------------------*/
 __global__ void hypre_BoomerAMGCreateS_dev1_mf( HYPRE_Int nr_of_rows, HYPRE_Real max_row_sum, HYPRE_Real strength_threshold,
					HYPRE_Real* A_diag_data, HYPRE_Int* A_diag_i, HYPRE_Int* A_diag_j,
					HYPRE_Real* A_offd_data, HYPRE_Int* A_offd_i, HYPRE_Int* A_offd_j,
                                        HYPRE_Int* S_temp_diag_j, HYPRE_Int* S_temp_offd_j,
					HYPRE_Int num_functions, HYPRE_Int* dof_func, HYPRE_Int* dof_func_offd,
					HYPRE_Int* jS_diag, HYPRE_Int* jS_offd )
{
   /*-----------------------------------------------------------------------*/
   /*
      Input: nr_of_rows - Number of rows in matrix (local in processor)
             A_diag_data, A_diag_i, A_diag_j - CSR representation of A_diag
             A_offd_data, A_offd_i, A_offd_j - CSR representation of A_offd
	     num_function  - Number of degrees of freedom per grid point
	     dof_func      - vector over nonzero elements of A_diag, indicating the degree of freedom
	     dof_func_offd - vector over nonzero elements of A_offd, indicating the degree of freedom

      Output: S_temp_diag_j - S_diag_j vector before compression, i.e.,elements that are -1 should be removed
              S_temp_offd_j - S_offd_j vector before compression, i.e.,elements that are -1 should be removed
	      jS_diag       - S_diag_i vector for compressed S_diag
	      jS_offd       - S_offd_i vector for compressed S_offd
    */
   /*-----------------------------------------------------------------------*/

   HYPRE_Real row_scale, row_sum, diag;
   HYPRE_Int i, myid = threadIdx.x + blockIdx.x * blockDim.x, jA, jSd, jSo, Adi, Adip, Aoi, Aoip;// cond;//, notallweak;
   const HYPRE_Int nthreads = gridDim.x * blockDim.x;

   for( i = myid ; i < nr_of_rows ; i += nthreads )
   {
      //      S_diag_i[i] = jS_diag;
      //      if (num_cols_offd)
      //      {
      //         S_offd_i[i] = jS_offd;
      //      }

      Adi = A_diag_i[i];
      Adip= A_diag_i[i+1];

      //      diag = A_diag_data[A_diag_i[i]];
      diag = A_diag_data[Adi];

      Aoi = A_offd_i[i];
      Aoip= A_offd_i[i+1];

 /* compute scaling factor and row sum */
      row_scale = 0.0;
      row_sum = diag;
      //      if (num_functions > 1)
      //      {
         if (diag < 0)
         {
            for (jA = Adi+1; jA < Adip; jA++)
            {
               if (dof_func[i] == dof_func[A_diag_j[jA]])
               {
                  row_scale = hypre_max(row_scale, A_diag_data[jA]);
                  row_sum += A_diag_data[jA];
               }
            }
            for (jA = Aoi; jA < Aoip; jA++)
            {
               if (dof_func[i] == dof_func_offd[A_offd_j[jA]])
               {
                  row_scale = hypre_max(row_scale, A_offd_data[jA]);
                  row_sum += A_offd_data[jA];
               }
            }
         }
         else
         {
            for (jA = Adi+1; jA < Adip; jA++)
            {
               if (dof_func[i] == dof_func[A_diag_j[jA]])
               {
                  row_scale = hypre_min(row_scale, A_diag_data[jA]);
                  row_sum += A_diag_data[jA];
               }
            }
            for (jA = Aoi; jA < Aoip; jA++)
            {
               if (dof_func[i] == dof_func_offd[A_offd_j[jA]])
               {
                  row_scale = hypre_min(row_scale, A_offd_data[jA]);
                  row_sum += A_offd_data[jA];
               }
            }
         } /* diag >= 0 */
	 //      } /* num_functions > 1 */

      //      jS_diag += A_diag_i[i + 1] - A_diag_i[i] - 1;
      //      jS_offd += A_offd_i[i + 1] - A_offd_i[i];

      //      jS_diag[i] = 0;
      //      jS_offd[i] = 0;
      jSd=jSo=0;
      /* compute row entries of S */
      S_temp_diag_j[Adi] = -1;
      //      notallweak = !((fabs(row_sum) > fabs(diag)*max_row_sum) && (max_row_sum < 1.0));
      if ((fabs(row_sum) > fabs(diag)*max_row_sum) && (max_row_sum < 1.0))
      {
         /* make all dependencies weak */
         for (jA = Adi+1; jA < Adip; jA++)
         {
            S_temp_diag_j[jA] = -1;
	 }
      	 //         jS_diag -= A_diag_i[i + 1] - (A_diag_i[i] + 1);

	 for (jA = Aoi; jA < Aoip; jA++)
	 {
	    S_temp_offd_j[jA] = -1;
	 }
      	 //         jS_offd -= A_offd_i[i + 1] - A_offd_i[i];
      }
      else
      {
	 //         if (num_functions > 1)
	 //         {
            if (diag < 0)
            {
               for (jA = Adi+1; jA < Adip; jA++)
               {
                  if (A_diag_data[jA] <= strength_threshold * row_scale
                      || dof_func[i] != dof_func[A_diag_j[jA]])
                  {
                     S_temp_diag_j[jA] = -1;
		     //                     --jS_diag;
                  }
                  else
                  {
                     S_temp_diag_j[jA] = A_diag_j[jA];
		     //		     jS_diag[i]++;
		     jSd++;
                  }
               }
               for (jA = Aoi; jA < Aoip; jA++)
               {
                  if (A_offd_data[jA] <= strength_threshold * row_scale
                      || dof_func[i] != dof_func_offd[A_offd_j[jA]])
                  {
                     S_temp_offd_j[jA] = -1;
		     //                     --jS_offd;
                  }
                  else
                  {
                     S_temp_offd_j[jA] = A_offd_j[jA];
		     //		     jS_offd[i]++;
		     jSo++;
                  }
               }
            }
            else
            {
               for (jA = Adi+1; jA < Adip; jA++)
               {
                  if (A_diag_data[jA] >= strength_threshold * row_scale
                      || dof_func[i] != dof_func[A_diag_j[jA]])
                  {
                     S_temp_diag_j[jA] = -1;
		     //                     --jS_diag;
                  }
                  else
                  {
                     S_temp_diag_j[jA] = A_diag_j[jA];
		     //		     jS_diag[i]++;
		     jSd++;
                  }
               }
               for (jA = Aoi; jA < Aoip; jA++)
               {
                  if (A_offd_data[jA] >= strength_threshold * row_scale
                      || dof_func[i] != dof_func_offd[A_offd_j[jA]])
                  {
                     S_temp_offd_j[jA] = -1;
		     //                     --jS_offd;
                  }
                  else
                  {
                     S_temp_offd_j[jA] = A_offd_j[jA];
		     //		     jS_offd[i]++;
		     jSo++;
                  }
               }
            } /* diag >= 0 */
      } /* !((row_sum > max_row_sum) && (max_row_sum < 1.0)) */
      jS_diag[i] = jSd;
      jS_offd[i] = jSo;
   } /* for each variable */
}

#endif /* #if defined(HYPRE_USING_CUDA) */

