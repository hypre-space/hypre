/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Matvec functions for hypre_CSRMatrix class.
 *
 *****************************************************************************/
#ifndef PAR_CSR_MATMULTIVEC_HEADER
#define PAR_CSR_MATMULTIVEC_HEADER

#include "parcsr_mv.h"
#include "par_multivector.h"

#ifdef __cplusplus
extern "C" {
#endif

int hypre_ParCSRMatrixMatMultiVec(double, hypre_ParCSRMatrix*,
                                  hypre_ParMultiVector*,
                                  double, hypre_ParMultiVector*);


int hypre_ParCSRMatrixMatMultiVecT(double, hypre_ParCSRMatrix*,
                                  hypre_ParMultiVector*,
                                  double, hypre_ParMultiVector*);

#ifdef __cplusplus
}
#endif

#endif  /* PAR_CSR_MATMULTIVEC_HEADER */
