/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/





#include "HYPRE_distributed_matrix_types.h"

#ifdef PETSC_AVAILABLE
/* HYPRE_ConvertPETScMatrixToDistributedMatrix.c */
HYPRE_Int HYPRE_ConvertPETScMatrixToDistributedMatrix (Mat PETSc_matrix , HYPRE_DistributedMatrix *DistributedMatrix );
#endif

/* HYPRE_ConvertParCSRMatrixToDistributedMatrix.c */
HYPRE_Int HYPRE_ConvertParCSRMatrixToDistributedMatrix (HYPRE_ParCSRMatrix parcsr_matrix , HYPRE_DistributedMatrix *DistributedMatrix );

