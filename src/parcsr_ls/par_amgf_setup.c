/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "par_amgf.h"
#include "_hypre_IJ_mv.h"

/*****************************************************************************
 *
 * Routine for driving the setup phase of AMGF
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * hypre_AMGFSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGFSetup(void * amgf_data_,
           hypre_ParCSRMatrix *A_,
           hypre_ParVector * b,
           hypre_ParVector * x)   
{
    hypre_ParAMGFData * amgf_data = (hypre_ParAMGFData *) amgf_data_;
    if (amgf_data->set_mask == 0)
    {
       printf("mask must be set prior to AMGFSetup call\n");
       exit(1);
    }
    if (amgf_data->set_coarse_solver == 0)
    {
       printf("CoarseSolver must be setup prior to AMGFSetup call\n");
       exit(1);
    }
    if (amgf_data->set_amg_solver == 0)
    {
       printf("AMG (smoother) must be setup prior to AMGFSetup call\n");
       exit(1);
    }
    HYPRE_ParCSRMatrix A = (HYPRE_ParCSRMatrix) A_; 
    MPI_Comm comm;
    HYPRE_ParCSRMatrixGetRowPartitioning(A, &(amgf_data->ipartition));
    HYPRE_ParCSRMatrixGetComm(A, &comm); 
    int myid, nprocs;
    MPI_Comm_rank(comm, &myid);
    MPI_Comm_size(comm, &nprocs);    
    
    HYPRE_BigInt ilower, iupper, local_size;
    HYPRE_BigInt jltemp, jutemp;
    HYPRE_ParCSRMatrixGetLocalRange(A, &ilower, &iupper, &jltemp, &jutemp);
    local_size = (iupper - ilower) + 1;

    amgf_data->ilower = ilower;
    amgf_data->iupper = iupper;
    amgf_data->local_size = local_size;

    HYPRE_BigInt cilower_loc = 0;
    HYPRE_BigInt ciupper_loc = -1;
    HYPRE_BigInt clocal_size = 0;
    for (int i = ilower; i <= iupper; i++)
    {
      if (amgf_data->constraint_mask[i-ilower] == 1)
      {
         ciupper_loc++;
      }
    }
    clocal_size = (ciupper_loc - cilower_loc) + 1;


    HYPRE_BigInt cilower, ciupper;
    cilower = 0; ciupper = 0;
    MPI_Status status;
    if (myid > 0)
    {
       int source_rank = myid - 1;
       int tag = myid - 1;
       MPI_Recv(&ciupper, 1, MPI_INT, source_rank, tag, comm, &status);
       ciupper += ciupper_loc;
    }
    else
    {
       ciupper = ciupper_loc;
    }
   
    if (myid < nprocs -1)
    {
       ciupper += 1;
       int send_rank = myid + 1;
       int tag = myid;
       MPI_Send(&ciupper, 1, MPI_INT, send_rank, tag, comm);
       ciupper -= 1;
    }
    cilower = ciupper - (clocal_size - 1);

    amgf_data->cilower = cilower;
    amgf_data->ciupper = ciupper;
    amgf_data->clocal_size = clocal_size;

    HYPRE_IJMatrixCreate(comm, cilower, ciupper, ilower, iupper, &(amgf_data->Rij));
    HYPRE_IJMatrixSetObjectType(amgf_data->Rij, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(amgf_data->Rij);
    {
       HYPRE_Int nrows = 1; /* set one row at a time */
       HYPRE_Int nnz = 1;   /* one nonzero per row in restriction matrix */
       double *values = (double *) malloc(nnz * sizeof(double)); values[0] = 1.0;
       HYPRE_BigInt *cols = (HYPRE_BigInt *) malloc(nnz * sizeof(HYPRE_BigInt));
       HYPRE_BigInt i = cilower;

       for (int j = 0; j < local_size; j++)
       {
          if (amgf_data->constraint_mask[j] == 1)
          {
             cols[0] = j + ilower;
             HYPRE_IJMatrixSetValues(amgf_data->Rij, nrows, &nnz, &i, cols, values);
             i++;
          }
       }
       free(values);
       free(cols);
    }
     
    HYPRE_IJMatrixAssemble(amgf_data->Rij);

    HYPRE_IJMatrixTranspose(amgf_data->Rij, &(amgf_data->Pij));
    HYPRE_IJMatrixAssemble(amgf_data->Pij);

    HYPRE_IJMatrixGetObject(amgf_data->Rij, (void**) &(amgf_data->R));
    HYPRE_IJMatrixGetObject(amgf_data->Pij, (void**) &(amgf_data->P));
    
    HYPRE_ParCSRMatrixGetGlobalRowPartitioning(amgf_data->R, 0, &(amgf_data->cpartition));
    HYPRE_ParCSRMatrixGetDims(amgf_data->R, &(amgf_data->global_m), &(amgf_data->global_n));

    /* determine the coarse operator*/
    amgf_data->Ac = hypre_ParCSRMatrixRAP(amgf_data->P, A, amgf_data->P);
    hypre_ParCSRMatrixSetNumNonzeros(amgf_data->Ac);
    
    // Setup work vector
    HYPRE_ParVectorCreate(amgf_data->comm, amgf_data->global_n, amgf_data->ipartition, &amgf_data->r);
    HYPRE_ParVectorCreate(amgf_data->comm, amgf_data->global_n, amgf_data->ipartition, &amgf_data->e);
    HYPRE_ParVectorCreate(amgf_data->comm, amgf_data->global_m, amgf_data->cpartition, &amgf_data->rc);
    HYPRE_ParVectorCreate(amgf_data->comm, amgf_data->global_m, amgf_data->cpartition, &amgf_data->ec);

    HYPRE_ParVectorInitialize(amgf_data->r);
    HYPRE_ParVectorInitialize(amgf_data->rc);
    HYPRE_ParVectorInitialize(amgf_data->ec);
    HYPRE_ParVectorInitialize(amgf_data->e);

    HYPRE_ParVectorSetConstantValues(amgf_data->r , 0.0);
    HYPRE_ParVectorSetConstantValues(amgf_data->rc, 0.0);
    HYPRE_ParVectorSetConstantValues(amgf_data->ec, 0.0);
    HYPRE_ParVectorSetConstantValues(amgf_data->e, 0.0);

   return hypre_error_flag;
};
