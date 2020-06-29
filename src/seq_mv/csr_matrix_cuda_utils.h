#ifndef CSR_MATRIX_CUDA_UTILS


#include "seq_mv.h"
#include <HYPRE_config.h>
#include "HYPRE_seq_mv.h"
#include "_hypre_utilities.hpp"
#include "HYPRE_utilities.h"
#include "csr_matrix.h"

#if defined(HYPRE_USING_CUDA)
#include <cusparse.h>



#if (CUDART_VERSION >= 8000)

cudaDataType_t hypre_getCudaDataTypeComplex();

void hypre_sortCSRCusparse(cusparseHandle_t cusparsehandle, HYPRE_Int n, HYPRE_Int m,
                   HYPRE_Int nnzA,
                   const HYPRE_Int *d_ia, HYPRE_Int *d_ja_sort, HYPRE_Complex *d_a_sort);

#endif
#if (CUDART_VERSION >= 10010)
cusparseSpMatDescr_t hypre_CSRMatToCuda(const hypre_CSRMatrix *A, HYPRE_Int offset);

cusparseDnVecDescr_t hypre_VecToCuda(const hypre_Vector *x, HYPRE_Int offset);


cusparseIndexType_t hypre_getCusparseIndexTypeInt() ;




cusparseSpMatDescr_t hypre_CSRMatRawToCuda(HYPRE_Int n, HYPRE_Int m, HYPRE_Int nnz, HYPRE_Int *i, HYPRE_Int *j, HYPRE_Complex *data);

#endif
#endif
#endif
