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
