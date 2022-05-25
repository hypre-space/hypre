/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * LoadBal.h header file.
 *
 *****************************************************************************/

#ifndef _LOADBAL_H
#define _LOADBAL_H

#define LOADBAL_REQ_TAG  888
#define LOADBAL_REP_TAG  889

typedef struct
{
    HYPRE_Int  pe;
    HYPRE_Int  beg_row;
    HYPRE_Int  end_row;
    HYPRE_Int *buffer;
}
DonorData;

typedef struct
{
    HYPRE_Int     pe;
    Matrix *mat;
    HYPRE_Real *buffer;
}
RecipData;

typedef struct
{
    HYPRE_Int         num_given;
    HYPRE_Int         num_taken;
    DonorData  *donor_data;
    RecipData  *recip_data;
    HYPRE_Int         beg_row;    /* local beginning row, after all donated rows */
}
LoadBal;

LoadBal *LoadBalDonate(MPI_Comm comm, Matrix *mat, Numbering *numb,
  HYPRE_Real local_cost, HYPRE_Real beta);
void LoadBalReturn(LoadBal *p, MPI_Comm comm, Matrix *mat);

#endif /* _LOADBAL_H */
