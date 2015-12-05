/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.7 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * Matvec functions for hypre_CSRMatrix class.
 *
 *****************************************************************************/
#ifndef PAR_CSR_MATMULTIVEC_HEADER
#define PAR_CSR_MATMULTIVEC_HEADER

#include "_hypre_parcsr_mv.h"
#include "par_multivector.h"

#ifdef __cplusplus
extern "C" {
#endif

HYPRE_Int hypre_ParCSRMatrixMatMultiVec(double, hypre_ParCSRMatrix*,
                                  hypre_ParMultiVector*,
                                  double, hypre_ParMultiVector*);


HYPRE_Int hypre_ParCSRMatrixMatMultiVecT(double, hypre_ParCSRMatrix*,
                                  hypre_ParMultiVector*,
                                  double, hypre_ParMultiVector*);

#ifdef __cplusplus
}
#endif

#endif  /* PAR_CSR_MATMULTIVEC_HEADER */
