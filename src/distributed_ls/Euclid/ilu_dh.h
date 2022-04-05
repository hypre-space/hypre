/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef ILU_MPI_DH
#define ILU_MPI_DH

/* #include "euclid_common.h" */

void reallocate_private(HYPRE_Int row, HYPRE_Int newEntries, HYPRE_Int *nzHave,
                HYPRE_Int **rp, HYPRE_Int **cval, float **aval, HYPRE_Real **avalD, HYPRE_Int **fill);

extern void ilu_mpi_pilu(Euclid_dh ctx);
  /* driver for comms intermingled with factorization */


extern void iluk_mpi_pilu(Euclid_dh ctx);
  /* the factorization algorithm */

extern void compute_scaling_private(HYPRE_Int row, HYPRE_Int len, HYPRE_Real *AVAL, Euclid_dh ctx);

extern void iluk_mpi_bj(Euclid_dh ctx);

extern void iluk_seq(Euclid_dh ctx);
extern void iluk_seq_block(Euclid_dh ctx);
  /* for sequential or parallel block jacobi.  If used
     for block jacobi, column indices are referenced to 0
     on return; make sure and add beg_row to these values
     before printing the matrix!

     1st version is for single precision, 2nd is for HYPRE_Real.
   */

extern void ilut_seq(Euclid_dh ctx);


#endif

