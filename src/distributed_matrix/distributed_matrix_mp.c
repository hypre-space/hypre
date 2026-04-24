/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"
#include "HYPRE_distributed_matrix_mv.h"

#if defined(HYPRE_MIXED_PRECISION)

#undef HYPRE_DistributedMatrixGetDims
#undef HYPRE_DistributedMatrixGetLocalRange
#undef HYPRE_DistributedMatrixGetRow
#undef HYPRE_DistributedMatrixRestoreRow

HYPRE_Int HYPRE_DistributedMatrixGetDims_flt(HYPRE_DistributedMatrix matrix, HYPRE_BigInt *M,
                                             HYPRE_BigInt *N);
HYPRE_Int HYPRE_DistributedMatrixGetDims_dbl(HYPRE_DistributedMatrix matrix, HYPRE_BigInt *M,
                                             HYPRE_BigInt *N);
HYPRE_Int HYPRE_DistributedMatrixGetDims_long_dbl(HYPRE_DistributedMatrix matrix, HYPRE_BigInt *M,
                                                  HYPRE_BigInt *N);

HYPRE_Int HYPRE_DistributedMatrixGetLocalRange_flt(HYPRE_DistributedMatrix matrix,
                                                   HYPRE_BigInt *row_start,
                                                   HYPRE_BigInt *row_end,
                                                   HYPRE_BigInt *col_start,
                                                   HYPRE_BigInt *col_end);
HYPRE_Int HYPRE_DistributedMatrixGetLocalRange_dbl(HYPRE_DistributedMatrix matrix,
                                                   HYPRE_BigInt *row_start,
                                                   HYPRE_BigInt *row_end,
                                                   HYPRE_BigInt *col_start,
                                                   HYPRE_BigInt *col_end);
HYPRE_Int HYPRE_DistributedMatrixGetLocalRange_long_dbl(HYPRE_DistributedMatrix matrix,
                                                        HYPRE_BigInt *row_start,
                                                        HYPRE_BigInt *row_end,
                                                        HYPRE_BigInt *col_start,
                                                        HYPRE_BigInt *col_end);

HYPRE_Int HYPRE_DistributedMatrixGetRow_flt(HYPRE_DistributedMatrix matrix, HYPRE_BigInt row,
                                            HYPRE_Int *size, HYPRE_BigInt **col_ind,
                                            hypre_float **values);
HYPRE_Int HYPRE_DistributedMatrixGetRow_dbl(HYPRE_DistributedMatrix matrix, HYPRE_BigInt row,
                                            HYPRE_Int *size, HYPRE_BigInt **col_ind,
                                            hypre_double **values);
HYPRE_Int HYPRE_DistributedMatrixGetRow_long_dbl(HYPRE_DistributedMatrix matrix,
                                                 HYPRE_BigInt row, HYPRE_Int *size,
                                                 HYPRE_BigInt **col_ind,
                                                 hypre_long_double **values);

HYPRE_Int HYPRE_DistributedMatrixRestoreRow_flt(HYPRE_DistributedMatrix matrix, HYPRE_BigInt row,
                                                HYPRE_Int *size, HYPRE_BigInt **col_ind,
                                                hypre_float **values);
HYPRE_Int HYPRE_DistributedMatrixRestoreRow_dbl(HYPRE_DistributedMatrix matrix, HYPRE_BigInt row,
                                                HYPRE_Int *size, HYPRE_BigInt **col_ind,
                                                hypre_double **values);
HYPRE_Int HYPRE_DistributedMatrixRestoreRow_long_dbl(HYPRE_DistributedMatrix matrix,
                                                     HYPRE_BigInt row, HYPRE_Int *size,
                                                     HYPRE_BigInt **col_ind,
                                                     hypre_long_double **values);

HYPRE_Int
HYPRE_DistributedMatrixGetDims(HYPRE_DistributedMatrix matrix, HYPRE_BigInt *M, HYPRE_BigInt *N)
{
   switch (hypre_GlobalPrecision())
   {
      case HYPRE_REAL_SINGLE:
         return HYPRE_DistributedMatrixGetDims_flt(matrix, M, N);
      case HYPRE_REAL_LONGDOUBLE:
         return HYPRE_DistributedMatrixGetDims_long_dbl(matrix, M, N);
      case HYPRE_REAL_DOUBLE:
      default:
         return HYPRE_DistributedMatrixGetDims_dbl(matrix, M, N);
   }
}

HYPRE_Int
HYPRE_DistributedMatrixGetLocalRange(HYPRE_DistributedMatrix matrix, HYPRE_BigInt *row_start,
                                     HYPRE_BigInt *row_end, HYPRE_BigInt *col_start,
                                     HYPRE_BigInt *col_end)
{
   switch (hypre_GlobalPrecision())
   {
      case HYPRE_REAL_SINGLE:
         return HYPRE_DistributedMatrixGetLocalRange_flt(matrix, row_start, row_end, col_start,
                                                         col_end);
      case HYPRE_REAL_LONGDOUBLE:
         return HYPRE_DistributedMatrixGetLocalRange_long_dbl(matrix, row_start, row_end,
                                                              col_start, col_end);
      case HYPRE_REAL_DOUBLE:
      default:
         return HYPRE_DistributedMatrixGetLocalRange_dbl(matrix, row_start, row_end, col_start,
                                                         col_end);
   }
}

HYPRE_Int
HYPRE_DistributedMatrixGetRow(HYPRE_DistributedMatrix matrix, HYPRE_BigInt row, HYPRE_Int *size,
                              HYPRE_BigInt **col_ind, HYPRE_Real **values)
{
   switch (hypre_GlobalPrecision())
   {
      case HYPRE_REAL_SINGLE:
         return HYPRE_DistributedMatrixGetRow_flt(matrix, row, size, col_ind,
                                                  (hypre_float **) values);
      case HYPRE_REAL_LONGDOUBLE:
         return HYPRE_DistributedMatrixGetRow_long_dbl(matrix, row, size, col_ind,
                                                       (hypre_long_double **) values);
      case HYPRE_REAL_DOUBLE:
      default:
         return HYPRE_DistributedMatrixGetRow_dbl(matrix, row, size, col_ind,
                                                  (hypre_double **) values);
   }
}

HYPRE_Int
HYPRE_DistributedMatrixRestoreRow(HYPRE_DistributedMatrix matrix, HYPRE_BigInt row,
                                  HYPRE_Int *size, HYPRE_BigInt **col_ind, HYPRE_Real **values)
{
   switch (hypre_GlobalPrecision())
   {
      case HYPRE_REAL_SINGLE:
         return HYPRE_DistributedMatrixRestoreRow_flt(matrix, row, size, col_ind,
                                                      (hypre_float **) values);
      case HYPRE_REAL_LONGDOUBLE:
         return HYPRE_DistributedMatrixRestoreRow_long_dbl(matrix, row, size, col_ind,
                                                           (hypre_long_double **) values);
      case HYPRE_REAL_DOUBLE:
      default:
         return HYPRE_DistributedMatrixRestoreRow_dbl(matrix, row, size, col_ind,
                                                      (hypre_double **) values);
   }
}

#endif
