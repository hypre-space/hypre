#ifndef CSR_MULTIMATVEC_H
#define CSR_MULTIMATVEC_H

#include "seq_mv.h"
#include "seq_multivector.h"

#ifdef __cplusplus
extern "C" {
#endif
/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMatMultivec
 *--------------------------------------------------------------------------*/
int
hypre_CSRMatrixMatMultivec(double alpha, hypre_CSRMatrix *A,
                           hypre_Multivector *x, double beta,
                           hypre_Multivector *y);
                            

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMultiMatvecT
 *
 *   Performs y <- alpha * A^T * x + beta * y
 *
 *   From Van Henson's modification of hypre_CSRMatrixMatvec.
 *--------------------------------------------------------------------------*/

int
hypre_CSRMatrixMatMultivecT(double alpha, hypre_CSRMatrix *A,
                            hypre_Multivector *x, double beta,
                            hypre_Multivector *y);
                             
#ifdef __cplusplus
}
#endif

#endif /* CSR_MATMULTIVEC_H */
