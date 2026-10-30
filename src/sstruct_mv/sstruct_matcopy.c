/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Routines that copy the structured component of a hypre_SStructMatrix (and,
 * optionally, its unstructured component) into ParCSR format on selected
 * boxes, and that compress the unstructured component back into the
 * structured one.
 *
 *****************************************************************************/

#include "_hypre_sstruct_mv.h"
#include "_hypre_struct_mv.hpp"
#include "_hypre_onedpl.hpp"

/* #define DEBUG_MATCONV */

/*--------------------------------------------------------------------------
 * Returns the stored entry of U at local row and global column col, or NULL
 *--------------------------------------------------------------------------*/

static inline
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
__host__ __device__
#endif
HYPRE_Complex *
hypre_SStructUMatrixFindEntry( HYPRE_Int       row,
                               HYPRE_BigInt    col,
                               HYPRE_BigInt    first_col,
                               HYPRE_BigInt    last_col,
                               HYPRE_Int      *di,
                               HYPRE_Int      *dj,
                               HYPRE_Complex  *da,
                               HYPRE_Int      *oi,
                               HYPRE_Int      *oj,
                               HYPRE_Complex  *oa,
                               HYPRE_BigInt   *col_map,
                               HYPRE_Int       num_cols_offd )
{
   HYPRE_Int  lo = 0;
   HYPRE_Int  hi = num_cols_offd - 1;
   HYPRE_Int  mid, j;

   if (col >= first_col && col <= last_col)
   {
      /* Diagonal block */
      for (j = di[row]; j < di[row + 1]; j++)
      {
         if (dj[j] == (HYPRE_Int) (col - first_col))
         {
            return da + j;
         }
      }
   }
   else if (oi[row] < oi[row + 1])
   {
      /* Off-diagonal block: binary search of the column in col_map */
      while (lo < hi)
      {
         mid = (lo + hi) / 2;
         if (col_map[mid] < col)
         {
            lo = mid + 1;
         }
         else
         {
            hi = mid;
         }
      }

      for (j = oi[row]; j < oi[row + 1] && col_map[lo] == col; j++)
      {
         if (oj[j] == lo)
         {
            return oa + j;
         }
      }
   }

   return NULL;
}

/*--------------------------------------------------------------------------
 * Extracts stored U entries: reads their values and sets them to zero in U.
 * Entries that are absent from U leave the corresponding values unchanged.
 * Unlike IJ GetValuesAndZeroOut, this implements the sparse overwrite needed
 * by U-to-S compression without replacing missing entries by zero.
 *
 * Row r of the request has ncols[r] columns, stored in cols and values from
 * position row_indexes[r].
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructUMatrixExtractValues( hypre_SStructMatrix *matrix,
                                   HYPRE_Int            nrows,
                                   HYPRE_Int           *ncols,
                                   HYPRE_BigInt        *rows,
                                   HYPRE_Int           *row_indexes,
                                   HYPRE_BigInt        *cols,
                                   HYPRE_Complex       *values )
{
   hypre_IJMatrix      *ij            = hypre_SStructMatrixIJMatrix(matrix);
   hypre_ParCSRMatrix  *U             = (hypre_ParCSRMatrix *) hypre_IJMatrixObject(ij);

   /* Diagonal block */
   hypre_CSRMatrix     *diag          = hypre_ParCSRMatrixDiag(U);
   HYPRE_Int           *di            = hypre_CSRMatrixI(diag);
   HYPRE_Int           *dj            = hypre_CSRMatrixJ(diag);
   HYPRE_Complex       *da            = hypre_CSRMatrixData(diag);
   HYPRE_Int            num_rows      = hypre_CSRMatrixNumRows(diag);
   HYPRE_BigInt         first_row     = hypre_ParCSRMatrixFirstRowIndex(U);
   HYPRE_BigInt         first_col     = hypre_ParCSRMatrixFirstColDiag(U);
   HYPRE_BigInt         last_col      = hypre_ParCSRMatrixLastColDiag(U);

   /* Off-diagonal block */
   hypre_CSRMatrix     *offd          = hypre_ParCSRMatrixOffd(U);
   HYPRE_Int           *oi            = hypre_CSRMatrixI(offd);
   HYPRE_Int           *oj            = hypre_CSRMatrixJ(offd);
   HYPRE_Complex       *oa            = hypre_CSRMatrixData(offd);
   HYPRE_BigInt        *col_map       = hypre_ParCSRMatrixColMapOffd(U);
   HYPRE_Int            num_cols_offd = hypre_CSRMatrixNumCols(offd);

#if defined(HYPRE_USING_GPU)
   if (hypre_GetExecPolicy1(hypre_ParCSRMatrixMemoryLocation(U)) == HYPRE_EXEC_DEVICE)
   {
      hypre_ParCSRMatrixCopyColMapOffdToDevice(U);
      col_map = hypre_ParCSRMatrixDeviceColMapOffd(U);
   }
#endif

   hypre_LoopBegin(nrows, r)
   {
      HYPRE_BigInt   local_row;
      HYPRE_Complex *u;
      HYPRE_Int      k;

      local_row = rows[r] - first_row;
      if (local_row >= 0 && local_row < num_rows)
      {
         for (k = row_indexes[r]; k < row_indexes[r] + ncols[r]; k++)
         {
            u = hypre_SStructUMatrixFindEntry((HYPRE_Int) local_row, cols[k], first_col,
                                              last_col, di, dj, da, oi, oj, oa, col_map,
                                              num_cols_offd);
            if (u)
            {
               values[k] = *u;
               *u = 0.0;
            }
         }
      }
   }
   hypre_LoopEnd();

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Rectangular piece of the structured component of a matrix, used to copy it
 * into (or compress an unstructured matrix back into) ParCSR format.
 *--------------------------------------------------------------------------*/

typedef struct
{
   HYPRE_Int       size[3], row_stride[3], col_stride[3], data_stride[3];
   HYPRE_Int       data_start, constant, volume, offset;
   HYPRE_BigInt    row_base, col_base;
   HYPRE_Complex  *data;
} hypre_SStructMatrixCopySegment;

/*--------------------------------------------------------------------------
 * Computes the local row, the global column and the position in the data of
 * the structured matrix of the k-th entry of segment s. Different entries of
 * the same segment always belong to different rows.
 *--------------------------------------------------------------------------*/

static inline
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
__host__ __device__
#endif
void
hypre_SStructSegmentEntry( const hypre_SStructMatrixCopySegment *s,
                           HYPRE_Int                             k,
                           HYPRE_BigInt                          first_row,
                           HYPRE_Int                            *row_ptr,
                           HYPRE_BigInt                         *col_ptr,
                           HYPRE_Int                            *data_ptr )
{
   HYPRE_Int  x = k % s->size[0];
   HYPRE_Int  y = (k / s->size[0]) % s->size[1];
   HYPRE_Int  z = k / (s->size[0] * s->size[1]);

   *row_ptr  = (HYPRE_Int) (s->row_base - first_row + (HYPRE_BigInt) x * s->row_stride[0] +
                            (HYPRE_BigInt) y * s->row_stride[1] +
                            (HYPRE_BigInt) z * s->row_stride[2]);
   *col_ptr  = s->col_base + (HYPRE_BigInt) x * s->col_stride[0] +
               (HYPRE_BigInt) y * s->col_stride[1] + (HYPRE_BigInt) z * s->col_stride[2];
   *data_ptr = s->constant ? 0 : s->data_start + x * s->data_stride[0] +
               y * s->data_stride[1] + z * s->data_stride[2];
}

/*--------------------------------------------------------------------------
 * Sets (add = 0) or adds (add = 1) value in column col of the row stored in
 * [start, *end) of cols and values, appending the column when absent. Repeated
 * structured entries are set, keeping the last one as IJ SetValues does.
 *--------------------------------------------------------------------------*/

static inline
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
__host__ __device__
#endif
void
hypre_SStructRowAccumulate( HYPRE_BigInt    *cols,
                            HYPRE_Complex   *values,
                            HYPRE_Int        start,
                            HYPRE_Int       *end,
                            HYPRE_BigInt     col,
                            HYPRE_Complex    value,
                            HYPRE_Int        add )
{
   HYPRE_Int  k = start;

   while (k < *end && cols[k] != col)
   {
      k++;
   }
   if (k == *end)
   {
      cols[(*end)++] = col;
      values[k] = 0.0;
   }
   values[k] = add ? values[k] + value : value;
}

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

/*--------------------------------------------------------------------------
 * Adds one to the number of entries of the rows of a segment. The rows of
 * different entries of a segment are different, so no atomics are needed as
 * long as the segments are processed one after the other.
 *--------------------------------------------------------------------------*/

struct hypre_SStructSegmentCountRows
{
   hypre_SStructMatrixCopySegment  s;
   HYPRE_BigInt                    first_row;
   HYPRE_Int                      *rowptr;

   __host__ __device__
   void operator()(HYPRE_Int k) const
   {
      HYPRE_Int     row, didx;
      HYPRE_BigInt  col;

      hypre_SStructSegmentEntry(&s, k, first_row, &row, &col, &didx);
      rowptr[row + 1]++;
   }
};

/*--------------------------------------------------------------------------
 * Appends the entries of a segment to their rows
 *--------------------------------------------------------------------------*/

struct hypre_SStructSegmentFillRows
{
   hypre_SStructMatrixCopySegment  s;
   HYPRE_BigInt                    first_row;
   HYPRE_Int                      *cursor;
   HYPRE_BigInt                   *cols;
   HYPRE_Complex                  *values;

   __host__ __device__
   void operator()(HYPRE_Int k) const
   {
      HYPRE_Int     row, didx, j;
      HYPRE_BigInt  col;

      hypre_SStructSegmentEntry(&s, k, first_row, &row, &col, &didx);
      j         = cursor[row]++;
      cols[j]   = col;
      values[j] = s.data[didx];
   }
};

/*--------------------------------------------------------------------------
 * Merges the repeated structured entries of each row (the last one wins, as in
 * IJ SetValues) and adds the entries of U. The entries of a row stay in the
 * order of their segments, so the rows and the summation order are repeatable.
 *--------------------------------------------------------------------------*/

struct hypre_SStructFinishRow
{
   HYPRE_BigInt    first_col;
   HYPRE_Int      *rowptr;
   HYPRE_Int      *cursor;
   HYPRE_BigInt   *cols;
   HYPRE_Complex  *values;
   HYPRE_Int      *di;
   HYPRE_Int      *dj;
   HYPRE_Complex  *da;
   HYPRE_Int      *oi;
   HYPRE_Int      *oj;
   HYPRE_Complex  *oa;
   HYPRE_BigInt   *col_map;

   __host__ __device__
   void operator()(HYPRE_Int row) const
   {
      HYPRE_Int  start = rowptr[row];
      HYPRE_Int  end   = start;
      HYPRE_Int  j;

      /* Merge repeated structured entries */
      for (j = start; j < cursor[row]; j++)
      {
         hypre_SStructRowAccumulate(cols, values, start, &end, cols[j], values[j], 0);
      }

      /* Add U (diagonal block first, then off-diagonal block) */
      if (di)
      {
         for (j = di[row]; j < di[row + 1]; j++)
         {
            hypre_SStructRowAccumulate(cols, values, start, &end, first_col + dj[j], da[j], 1);
         }
         for (j = oi[row]; j < oi[row + 1]; j++)
         {
            hypre_SStructRowAccumulate(cols, values, start, &end, col_map[oj[j]], oa[j], 1);
         }
      }

      cursor[row] = end;
   }
};

/*--------------------------------------------------------------------------
 * Number of entries of a row of U (diagonal plus off-diagonal blocks)
 *--------------------------------------------------------------------------*/

struct hypre_SStructURowSize
{
   HYPRE_Int  *di;
   HYPRE_Int  *oi;

   __host__ __device__
   HYPRE_Int operator()(HYPRE_Int row) const
   {
      return (di[row + 1] - di[row]) + (oi[row + 1] - oi[row]);
   }
};

/*--------------------------------------------------------------------------
 * Number of entries of a row, given the positions where it ends and starts
 *--------------------------------------------------------------------------*/

struct hypre_SStructRowLength
{
   __host__ __device__
   HYPRE_Int operator()(HYPRE_Int end, HYPRE_Int start) const
   {
      return end - start;
   }
};

/*--------------------------------------------------------------------------
 * Flags the entries that remain in each row after merging
 *--------------------------------------------------------------------------*/

struct hypre_SStructFlagRowEntries
{
   HYPRE_Int  *rowptr;
   HYPRE_Int  *cursor;
   HYPRE_Int  *flags;

   __host__ __device__
   void operator()(HYPRE_Int row) const
   {
      HYPRE_Int  j;

      for (j = rowptr[row]; j < cursor[row]; j++)
      {
         flags[j] = 1;
      }
   }
};

/*--------------------------------------------------------------------------
 * Predicate for flagged entries
 *--------------------------------------------------------------------------*/

struct hypre_SStructIsFlagged
{
   __host__ __device__
   bool operator()(HYPRE_Int flag) const
   {
      return flag != 0;
   }
};

/*--------------------------------------------------------------------------
 * Device CSR assembly: count the entries of each row, scan, and then fill
 * the rows with the entries of the segments and of U
 *--------------------------------------------------------------------------*/

static void
hypre_SStructMatrixCopyBoxesDevice( hypre_IJMatrix                 *ij,
                                    hypre_SStructMatrixCopySegment *segments,
                                    HYPRE_Int                       count,
                                    HYPRE_Int                       nnz,
                                    hypre_ParCSRMatrix             *U )
{
   HYPRE_BigInt                     range[4];
   HYPRE_Int                        nrows, num_cols_offd, new_nnz, e;

   hypre_ParCSRMatrix              *par;
   hypre_CSRMatrix                 *csr, *diag, *offd;
   hypre_CSRMatrix                 *ud, *uo;
   HYPRE_Int                       *rowptr, *cursor, *map, *new_i, *flags;
   HYPRE_BigInt                    *cols, *new_j, *colmap;
   HYPRE_Complex                   *values, *new_data;
   HYPRE_Int                       *di = NULL, *dj = NULL, *oi = NULL, *oj = NULL;
   HYPRE_Complex                   *da = NULL, *oa = NULL;
   HYPRE_BigInt                    *umap = NULL;

   if (U)
   {
      nnz += hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(U)) +
             hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(U));
   }
   HYPRE_IJMatrixGetLocalRange(ij, range, range + 1, range + 2, range + 3);
   nrows = (HYPRE_Int) (range[1] - range[0] + 1);

   /* Allocate memory */
   csr    = hypre_CSRMatrixCreate(nrows, 0, nnz);
   rowptr = hypre_CTAlloc(HYPRE_Int, nrows + 1, HYPRE_MEMORY_DEVICE);
   cursor = hypre_TAlloc(HYPRE_Int, nrows, HYPRE_MEMORY_DEVICE);
   cols   = hypre_TAlloc(HYPRE_BigInt, nnz, HYPRE_MEMORY_DEVICE);
   values = hypre_TAlloc(HYPRE_Complex, nnz, HYPRE_MEMORY_DEVICE);

   /* Count the entries of each row */
   if (U)
   {
      ud = hypre_ParCSRMatrixDiag(U);
      uo = hypre_ParCSRMatrixOffd(U);
      di = hypre_CSRMatrixI(ud);
      dj = hypre_CSRMatrixJ(ud);
      da = hypre_CSRMatrixData(ud);
      oi = hypre_CSRMatrixI(uo);
      oj = hypre_CSRMatrixJ(uo);
      oa = hypre_CSRMatrixData(uo);
      hypre_ParCSRMatrixCopyColMapOffdToDevice(U);
      umap = hypre_ParCSRMatrixDeviceColMapOffd(U);

      hypre_SStructURowSize  u_size = {di, oi};

      HYPRE_THRUST_CALL( transform,
                         thrust::make_counting_iterator(0),
                         thrust::make_counting_iterator(nrows),
                         rowptr + 1,
                         u_size );
   }
   for (e = 0; e < count; e++)
   {
      hypre_SStructSegmentCountRows  count_rows = {segments[e], range[0], rowptr};

      HYPRE_THRUST_CALL( for_each,
                         thrust::make_counting_iterator(0),
                         thrust::make_counting_iterator(segments[e].volume),
                         count_rows );
   }
   hypre_IntegerInclusiveScanDevice(nrows + 1, rowptr);
   hypre_TMemcpy(cursor, rowptr, HYPRE_Int, nrows, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

   /* Fill the rows: first the structured entries, in the order of the segments */
   for (e = 0; e < count; e++)
   {
      hypre_SStructSegmentFillRows  fill_rows = {segments[e], range[0], cursor, cols, values};

      HYPRE_THRUST_CALL( for_each,
                         thrust::make_counting_iterator(0),
                         thrust::make_counting_iterator(segments[e].volume),
                         fill_rows );
   }

   /* Merge repeated entries and add U */
   hypre_SStructFinishRow  finish_row = {range[2], rowptr, cursor, cols, values,
                                         di, dj, da, oi, oj, oa, umap
                                        };
   HYPRE_THRUST_CALL( for_each,
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(nrows),
                      finish_row );

   /* Pack the rows if there were repeated entries */
   new_i = hypre_CTAlloc(HYPRE_Int, nrows + 1, HYPRE_MEMORY_DEVICE);
   HYPRE_THRUST_CALL( transform, cursor, cursor + nrows, rowptr, new_i + 1,
                      hypre_SStructRowLength() );
   hypre_IntegerInclusiveScanDevice(nrows + 1, new_i);
   hypre_TMemcpy(&new_nnz, new_i + nrows, HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   if (new_nnz != nnz)
   {
      hypre_SStructFlagRowEntries  flag_entries = {rowptr, cursor, NULL};

      flags = hypre_CTAlloc(HYPRE_Int, nnz, HYPRE_MEMORY_DEVICE);
      flag_entries.flags = flags;
      HYPRE_THRUST_CALL( for_each,
                         thrust::make_counting_iterator(0),
                         thrust::make_counting_iterator(nrows),
                         flag_entries );

      new_j    = hypre_TAlloc(HYPRE_BigInt, new_nnz, HYPRE_MEMORY_DEVICE);
      new_data = hypre_TAlloc(HYPRE_Complex, new_nnz, HYPRE_MEMORY_DEVICE);
      HYPRE_THRUST_CALL( copy_if, cols, cols + nnz, flags, new_j, hypre_SStructIsFlagged() );
      HYPRE_THRUST_CALL( copy_if, values, values + nnz, flags, new_data,
                         hypre_SStructIsFlagged() );
      hypre_TFree(flags, HYPRE_MEMORY_DEVICE);
      hypre_TFree(cols, HYPRE_MEMORY_DEVICE);
      hypre_TFree(values, HYPRE_MEMORY_DEVICE);

      cols   = new_j;
      values = new_data;
      nnz    = new_nnz;
      hypre_TFree(rowptr, HYPRE_MEMORY_DEVICE);
      rowptr = new_i;
   }
   else
   {
      hypre_TFree(new_i, HYPRE_MEMORY_DEVICE);
   }

   /* Split the CSR matrix into diag and offd and set up the ParCSR matrix */
   hypre_CSRMatrixNumNonzeros(csr)    = nnz;
   hypre_CSRMatrixMemoryLocation(csr) = HYPRE_MEMORY_DEVICE;
   hypre_CSRMatrixI(csr)              = rowptr;
   hypre_CSRMatrixBigJ(csr)           = cols;
   hypre_CSRMatrixData(csr)           = values;
   hypre_CSRMatrixSplitDevice(csr, range[2], range[3], 0, NULL, &map,
                              &num_cols_offd, &colmap, &diag, &offd);

   hypre_IJMatrixCreateParCSR(ij);
   par = (hypre_ParCSRMatrix *) hypre_IJMatrixObject(ij);
   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(par));
   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffd(par));
   hypre_ParCSRMatrixDiag(par)             = diag;
   hypre_ParCSRMatrixOffd(par)             = offd;
   hypre_ParCSRMatrixDeviceColMapOffd(par) = colmap;
   hypre_ParCSRMatrixColMapOffd(par)       = hypre_TAlloc(HYPRE_BigInt, num_cols_offd,
                                                          HYPRE_MEMORY_HOST);
   hypre_TMemcpy(hypre_ParCSRMatrixColMapOffd(par), colmap, HYPRE_BigInt, num_cols_offd,
                 HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   hypre_CSRMatrixSetRownnz(diag);
   hypre_CSRMatrixSetRownnz(offd);
   hypre_ParCSRMatrixSetNumNonzeros(par);
   hypre_IJMatrixAssembleFlag(ij) = 1;

   /* Free memory */
   hypre_CSRMatrixDestroy(csr);
   hypre_TFree(map, HYPRE_MEMORY_DEVICE);
   hypre_TFree(cursor, HYPRE_MEMORY_DEVICE);
}
#endif

/*--------------------------------------------------------------------------
 * Host CSR assembly: structured and U coefficients share one pass
 *--------------------------------------------------------------------------*/

static void
hypre_SStructMatrixCopyBoxesHost( hypre_IJMatrix                 *ij,
                                  hypre_SStructMatrixCopySegment *segments,
                                  HYPRE_Int                       count,
                                  HYPRE_Int                       nnz,
                                  hypre_ParCSRMatrix             *U )
{
   hypre_CSRMatrix                 *ud = U ? hypre_ParCSRMatrixDiag(U) : NULL;
   hypre_CSRMatrix                 *uo = U ? hypre_ParCSRMatrixOffd(U) : NULL;
   hypre_SStructMatrixCopySegment  *s;
   hypre_ParCSRMatrix              *par;
   hypre_CSRMatrix                 *csr, *diag, *offd, *ucsr;
   HYPRE_BigInt                     range[4];
   HYPRE_BigInt                    *colmap, *jj;
   HYPRE_BigInt                     col;
   HYPRE_Complex                   *data;
   HYPRE_Int                       *ii, *cursor;
   HYPRE_Int                        nrows, noffd, packed, start, end;
   HYPRE_Int                        e, r, x, y, z, j, block, row, ucol;

   HYPRE_IJMatrixGetLocalRange(ij, range, range + 1, range + 2, range + 3);
   nrows = (HYPRE_Int) (range[1] - range[0] + 1);
   ii    = hypre_CTAlloc(HYPRE_Int, nrows + 1, HYPRE_MEMORY_HOST);

   /* Count the entries of each row */
   if (U)
   {
      nnz += hypre_CSRMatrixNumNonzeros(ud) + hypre_CSRMatrixNumNonzeros(uo);
      for (r = 0; r < nrows; r++)
      {
         ii[r + 1] = hypre_CSRMatrixI(ud)[r + 1] - hypre_CSRMatrixI(ud)[r] +
                     hypre_CSRMatrixI(uo)[r + 1] - hypre_CSRMatrixI(uo)[r];
      }
   }
   for (e = 0; e < count; e++)
   {
      s = segments + e;
      for (z = 0; z < s->size[2]; z++)
      {
         for (y = 0; y < s->size[1]; y++)
         {
            row = (HYPRE_Int) (s->row_base - range[0] + (HYPRE_BigInt) y * s->row_stride[1] +
                               (HYPRE_BigInt) z * s->row_stride[2]);
            for (x = 0; x < s->size[0]; x++)
            {
               ii[row + x * s->row_stride[0] + 1]++;
            }
         }
      }
   }
   for (r = 0; r < nrows; r++)
   {
      ii[r + 1] += ii[r];
   }

   /* Fill the rows with the entries of the segments */
   cursor = hypre_TAlloc(HYPRE_Int, nrows, HYPRE_MEMORY_HOST);
   hypre_TMemcpy(cursor, ii, HYPRE_Int, nrows, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
   jj   = hypre_TAlloc(HYPRE_BigInt, nnz, HYPRE_MEMORY_HOST);
   data = hypre_TAlloc(HYPRE_Complex, nnz, HYPRE_MEMORY_HOST);
   for (e = 0; e < count; e++)
   {
      s = segments + e;
      for (z = 0; z < s->size[2]; z++)
      {
         for (y = 0; y < s->size[1]; y++)
         {
            row   = (HYPRE_Int) (s->row_base - range[0] + (HYPRE_BigInt) y * s->row_stride[1] +
                                 (HYPRE_BigInt) z * s->row_stride[2]);
            col   = s->col_base + (HYPRE_BigInt) y * s->col_stride[1] +
                    (HYPRE_BigInt) z * s->col_stride[2];
            start = s->data_start + y * s->data_stride[1] + z * s->data_stride[2];
            for (x = 0; x < s->size[0]; x++)
            {
               j = cursor[row + x * s->row_stride[0]]++;
               jj[j]   = col + (HYPRE_BigInt) x * s->col_stride[0];
               data[j] = s->data[s->constant ? 0 : start + x * s->data_stride[0]];
            }
         }
      }
   }

   /* Merge repeated structured entries and add U */
   for (r = 0; r < nrows; r++)
   {
      end = ii[r];
      for (j = ii[r]; j < cursor[r]; j++)
      {
         hypre_SStructRowAccumulate(jj, data, ii[r], &end, jj[j], data[j], 0);
      }

      for (block = 0; U && block < 2; block++)
      {
         ucsr = block ? uo : ud;
         for (j = hypre_CSRMatrixI(ucsr)[r]; j < hypre_CSRMatrixI(ucsr)[r + 1]; j++)
         {
            ucol = hypre_CSRMatrixJ(ucsr)[j];
            hypre_SStructRowAccumulate(jj, data, ii[r], &end,
                                       block ? hypre_ParCSRMatrixColMapOffd(U)[ucol] :
                                       range[2] + ucol,
                                       hypre_CSRMatrixData(ucsr)[j], 1);
         }
      }
      cursor[r] = end;
   }

   /* Pack the rows */
   packed = 0;
   for (r = 0; r < nrows; r++)
   {
      start = ii[r];
      ii[r] = packed;
      for (j = start; j < cursor[r]; j++)
      {
         jj[packed]     = jj[j];
         data[packed++] = data[j];
      }
   }
   ii[nrows] = packed;

   /* Split the CSR matrix into diag and offd and set up the ParCSR matrix */
   csr = hypre_CSRMatrixCreate(nrows, 0, packed);
   hypre_CSRMatrixMemoryLocation(csr) = HYPRE_MEMORY_HOST;
   hypre_CSRMatrixI(csr)              = ii;
   hypre_CSRMatrixBigJ(csr)           = jj;
   hypre_CSRMatrixData(csr)           = data;
   hypre_CSRMatrixSplit(csr, range[2], range[3], 0, NULL, &noffd, &colmap, &diag, &offd);

   hypre_IJMatrixCreateParCSR(ij);
   par = (hypre_ParCSRMatrix *) hypre_IJMatrixObject(ij);
   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(par));
   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffd(par));
   hypre_ParCSRMatrixDiag(par)       = diag;
   hypre_ParCSRMatrixOffd(par)       = offd;
   hypre_ParCSRMatrixColMapOffd(par) = colmap;
   hypre_CSRMatrixSetRownnz(diag);
   hypre_CSRMatrixSetRownnz(offd);
   hypre_ParCSRMatrixSetNumNonzeros(par);
   hypre_IJMatrixAssembleFlag(ij) = 1;

   /* Free memory */
   hypre_CSRMatrixDestroy(csr);
   hypre_TFree(cursor, HYPRE_MEMORY_HOST);
}

/* Reject unsupported layouts before collecting support or constructing
 * segments. Compression additionally requires a square, nonconstant stencil. */
static HYPRE_Int
hypre_SStructMatrixCopyLayout( hypre_SStructMatrix *A,
                               hypre_SStructGrid   *grid,
                               HYPRE_Int            compress )
{
   HYPRE_Int              ndim   = hypre_SStructMatrixNDim(A);
   HYPRE_Int              nparts = hypre_SStructMatrixNParts(A);
   hypre_SStructGraph    *graph  = hypre_SStructMatrixGraph(A);

   hypre_SStructPMatrix  *pA;
   hypre_SStructPGrid    *pgrid;
   hypre_SStructStencil  *stencil;
   hypre_StructMatrix    *sA;
   hypre_IndexRef         stride;
   HYPRE_Int             *split;
   HYPRE_Int             *smap;
   HYPRE_Int              part, d, e, se;

   if (ndim < 1 || ndim > 3 ||
       hypre_SStructGridNParts(grid) != nparts ||
       (compress && hypre_SStructGraphDomGrid(graph) != grid))
   {
      return 0;
   }

   for (part = 0; part < nparts; part++)
   {
      pA    = hypre_SStructMatrixPMatrix(A, part);
      pgrid = hypre_SStructGridPGrid(grid, part);

      if (hypre_SStructPMatrixNVars(pA) != 1 ||
          hypre_SStructPGridNVars(pgrid) != 1 ||
          hypre_SStructPGridVarType(pgrid, 0) != HYPRE_SSTRUCT_VARIABLE_CELL)
      {
         return 0;
      }

      sA = hypre_SStructPMatrixSMatrix(pA, 0, 0);
      if (!sA)
      {
         return 0;
      }

      stride = hypre_SStructPMatrixDomainStride(pA);
      for (d = 0; d < ndim; d++)
      {
         if (stride[d] < 1 ||
             hypre_SStructPMatrixRangeStride(pA)[d] != 1 ||
             hypre_StructMatrixDataStride(sA)[d] != stride[d] ||
             (compress && stride[d] != 1))
         {
            return 0;
         }
      }

      stencil = hypre_SStructGraphStencil(graph, part, 0);
      split   = hypre_SStructMatrixSplit(A, part, 0);
      smap    = hypre_SStructPMatrixSMap(pA, 0);
      for (e = 0; e < hypre_SStructStencilSize(stencil); e++)
      {
         if (split[e] < 0 || hypre_SStructStencilVars(stencil)[e] != 0)
         {
            return 0;
         }

         se = smap[split[e]];
         if (hypre_StructMatrixSymmEntries(sA)[se] >= 0 ||
             (compress && hypre_StructMatrixConstEntry(sA, se)))
         {
            return 0;
         }
      }
   }

   return 1;
}

/*--------------------------------------------------------------------------
 * Builds the segments of the structured component of A over the rows in boxes:
 * one segment per stencil entry and box manager intersection of rows and
 * columns. col_action selects the column box managers as in
 * hypre_SStructGridIntersect. The layout must pass hypre_SStructMatrixCopyLayout.
 * Returns zero (locally) for selections outside the local boxes or for too many
 * entries.
 *--------------------------------------------------------------------------*/

static HYPRE_Int
hypre_SStructMatrixBoxSegments( hypre_SStructMatrix              *A,
                                hypre_SStructGrid                *grid,
                                hypre_BoxArray                ***boxes,
                                HYPRE_Int                        col_action,
                                hypre_SStructMatrixCopySegment  **segments_ptr,
                                HYPRE_Int                        *count_ptr,
                                HYPRE_Int                        *nnz_ptr )
{
   HYPRE_Int                        ndim     = hypre_SStructMatrixNDim(A);
   HYPRE_Int                        nparts   = hypre_SStructMatrixNParts(A);
   HYPRE_Int                        type     = hypre_SStructMatrixObjectType(A);
   hypre_SStructGraph              *graph    = hypre_SStructMatrixGraph(A);
   hypre_SStructGrid               *dom_grid = hypre_SStructGraphDomGrid(graph);

   /* Segments */
   hypre_SStructMatrixCopySegment  *segments = NULL;
   hypre_SStructMatrixCopySegment   s;
   HYPRE_Int                        count    = 0;
   HYPRE_Int                        capacity = 0;
   HYPRE_Int                        nnz      = 0;
   HYPRE_Int                        valid    = 1;

   /* Matrix data of each part */
   hypre_SStructPMatrix            *pA;
   hypre_SStructPGrid              *pgrid;
   hypre_SStructStencil            *stencil;
   hypre_StructMatrix              *sA;
   hypre_BoxArray                  *grid_boxes;
   hypre_IndexRef                   stride;
   hypre_IndexRef                   offset;
   HYPRE_Int                       *split;
   HYPRE_Int                       *smap;

   /* Boxes and box manager entries */
   hypre_BoxArray                  *outside, *scratch, *selected;
   hypre_Box                       *box, *data_box;
   hypre_Box                       *clipped, *region, *map_box, *to_box, *intersection;
   hypre_BoxManEntry              **row_entries;
   hypre_BoxManEntry              **col_entries;
   hypre_SStructBoxManInfo         *from_info, *to_info;
   hypre_Index                      rs, cs, data_index;
   HYPRE_Int                        nrow_entries, ncol_entries;
   HYPRE_Int                        from_part, to_part;
   HYPRE_Int                        data_step, sentry;
   HYPRE_Int                        part, b, g, e, i, j, d;

   clipped      = hypre_BoxCreate(ndim);
   region       = hypre_BoxCreate(ndim);
   map_box      = hypre_BoxCreate(ndim);
   to_box       = hypre_BoxCreate(ndim);
   intersection = hypre_BoxCreate(ndim);

   for (part = 0; valid && part < nparts; part++)
   {
      pA         = hypre_SStructMatrixPMatrix(A, part);
      pgrid      = hypre_SStructGridPGrid(grid, part);
      sA         = hypre_SStructPMatrixSMatrix(pA, 0, 0);
      stride     = hypre_SStructPMatrixDomainStride(pA);
      stencil    = hypre_SStructGraphStencil(graph, part, 0);
      split      = hypre_SStructMatrixSplit(A, part, 0);
      smap       = hypre_SStructPMatrixSMap(pA, 0);
      grid_boxes = hypre_StructGridBoxes(hypre_SStructPGridSGrid(pgrid, 0));

      /* Clipping must not discard selected ghost rows. The general backend
         handles selections outside the owned boxes. Selections spanning more
         than one owned box are handled by the per-source union below. */
      outside = hypre_BoxArrayClone(boxes[part][0]);
      scratch = hypre_BoxArrayCreate(0, ndim);
      hypre_SubtractBoxArrays(outside, grid_boxes, scratch);
      valid = (hypre_BoxArraySize(outside) == 0);
      hypre_BoxArrayDestroy(outside);
      hypre_BoxArrayDestroy(scratch);
      if (!valid)
      {
         break;
      }

      selected = hypre_BoxArrayCreate(0, ndim);
      for (g = 0; valid && g < hypre_BoxArraySize(grid_boxes); g++)
      {
         hypre_BoxArraySetSize(selected, 0);
         hypre_ForBoxI(b, boxes[part][0])
         {
            hypre_IntersectBoxes(hypre_BoxArrayBox(boxes[part][0], b),
                                 hypre_BoxArrayBox(grid_boxes, g), clipped);
            if (hypre_BoxVolume(clipped))
            {
               hypre_AppendBox(clipped, selected);
            }
         }

         /* Union within each source box so overlapping selections neither
            duplicate coefficients nor force a slow IJ assembly */
         hypre_UnionBoxes(selected);

         data_box = hypre_StructMatrixBoxDataBox(sA, g);
         for (b = 0; valid && b < hypre_BoxArraySize(selected); b++)
         {
            box = hypre_BoxArrayBox(selected, b);
            hypre_SStructGridIntersect(grid, part, 0, box, -1, &row_entries, &nrow_entries);
            for (i = 0; valid && i < nrow_entries; i++)
            {
               hypre_SStructBoxManEntryGetStrides(row_entries[i], rs, type);
               hypre_BoxManEntryGetExtents(row_entries[i], hypre_BoxIMin(map_box),
                                           hypre_BoxIMax(map_box));
               hypre_IntersectBoxes(box, map_box, region);

               for (e = 0; valid && e < hypre_SStructStencilSize(stencil); e++)
               {
                  offset = hypre_SStructStencilShape(stencil)[e];
                  sentry = smap[split[e]];

                  hypre_CopyBox(region, to_box);
                  hypre_BoxShiftPos(to_box, offset);
                  hypre_CoarsenBox(to_box, NULL, stride);
                  hypre_SStructGridIntersect(dom_grid, part, 0, to_box, col_action,
                                             &col_entries, &ncol_entries);
                  for (j = 0; valid && j < ncol_entries; j++)
                  {
                     /* Skip the couplings within the same part (and box manager
                        info type) when looking for the inter-part ones */
                     if (col_action == 1)
                     {
                        hypre_SStructBoxManEntryGetPart(row_entries[i], part, &from_part);
                        hypre_SStructBoxManEntryGetPart(col_entries[j], part, &to_part);
                        hypre_BoxManEntryGetInfo(row_entries[i], (void **) &from_info);
                        hypre_BoxManEntryGetInfo(col_entries[j], (void **) &to_info);
                        if (from_part == to_part &&
                            hypre_SStructBoxManInfoType(from_info) ==
                            hypre_SStructBoxManInfoType(to_info))
                        {
                           continue;
                        }
                     }

                     hypre_BoxManEntryGetExtents(col_entries[j], hypre_BoxIMin(map_box),
                                                 hypre_BoxIMax(map_box));
                     hypre_IntersectBoxes(to_box, map_box, intersection);
                     if (!hypre_BoxVolume(intersection))
                     {
                        continue;
                     }

                     hypre_SStructBoxManEntryGetStrides(col_entries[j], cs, type);
                     hypre_SStructBoxManEntryGetGlobalRank(col_entries[j],
                                                           hypre_BoxIMin(intersection),
                                                           &s.col_base, type);
                     s.volume = hypre_BoxVolume(intersection);
                     if (s.volume > HYPRE_INT_MAX - nnz)
                     {
                        valid = 0;
                        break;
                     }

                     data_step = 1;
                     for (d = 0; d < 3; d++)
                     {
                        s.size[d]        = d < ndim ? hypre_BoxSizeD(intersection, d) : 1;
                        s.row_stride[d]  = d < ndim ? rs[d] * stride[d] : 0;
                        s.col_stride[d]  = d < ndim ? cs[d] : 0;
                        s.data_stride[d] = d < ndim ? data_step : 0;
                        if (d < ndim)
                        {
                           data_step *= hypre_BoxSizeD(data_box, d);
                        }
                     }

                     hypre_RefineBox(intersection, NULL, stride);
                     hypre_BoxShiftNeg(intersection, offset);
                     hypre_SStructBoxManEntryGetGlobalRank(row_entries[i],
                                                           hypre_BoxIMin(intersection),
                                                           &s.row_base, type);
                     hypre_CopyIndex(hypre_BoxIMin(intersection), data_index);
                     hypre_StructMatrixMapDataIndex(sA, data_index);
                     s.data_start = hypre_BoxIndexRank(data_box, data_index);
                     s.constant   = hypre_StructMatrixConstEntry(sA, sentry);
                     s.data       = hypre_StructMatrixBoxData(sA, g, sentry);
                     s.offset     = nnz;
                     nnz         += s.volume;

                     if (count == capacity)
                     {
                        capacity = capacity > HYPRE_INT_MAX / 2 ? HYPRE_INT_MAX :
                                   hypre_max(16, 2 * capacity);
                        segments = hypre_TReAlloc(segments, hypre_SStructMatrixCopySegment,
                                                  capacity, HYPRE_MEMORY_HOST);
                     }
                     segments[count++] = s;
                  }
                  hypre_TFree(col_entries, HYPRE_MEMORY_HOST);
               }
            }
            hypre_TFree(row_entries, HYPRE_MEMORY_HOST);
         }
      }
      hypre_BoxArrayDestroy(selected);
   }

   /* Free memory */
   hypre_BoxDestroy(clipped);
   hypre_BoxDestroy(region);
   hypre_BoxDestroy(map_box);
   hypre_BoxDestroy(to_box);
   hypre_BoxDestroy(intersection);

   *segments_ptr = segments;
   *count_ptr    = count;
   *nnz_ptr      = nnz;

   return valid;
}

/*--------------------------------------------------------------------------
 * Assembles the structured component of A on the rows in boxes (plus U, if
 * not NULL) directly as a ParCSR matrix. Returns zero on all ranks when the
 * layout is not supported or when result is already initialized.
 *--------------------------------------------------------------------------*/

static HYPRE_Int
hypre_SStructMatrixCopyBoxes( hypre_SStructMatrix  *A,
                              hypre_SStructGrid    *grid,
                              hypre_IJMatrix      **result,
                              hypre_BoxArray     ***boxes,
                              hypre_ParCSRMatrix   *U )
{
   MPI_Comm                         comm = hypre_SStructMatrixComm(A);
   HYPRE_IJMatrix                   ij_A = hypre_SStructMatrixIJMatrix(A);
   HYPRE_IJMatrix                   ij;
   hypre_SStructMatrixCopySegment  *segments = NULL;
   HYPRE_Int                        count    = 0;
   HYPRE_Int                        nnz      = 0;
   HYPRE_Int                        diag_nnz, offd_nnz;
   HYPRE_Int                        valid, all_valid;
#if defined(HYPRE_USING_GPU)
   HYPRE_MemoryLocation             location = hypre_IJMatrixMemoryLocation(ij_A);
#endif

   valid = !*result &&
           !hypre_IJMatrixGlobalFirstRow(ij_A) &&
           !hypre_IJMatrixGlobalFirstCol(ij_A) &&
           hypre_SStructMatrixCopyLayout(A, grid, 0);
#if defined(HYPRE_USING_GPU) && !defined(HYPRE_USING_CUDA) && !defined(HYPRE_USING_HIP)
   if (hypre_GetExecPolicy1(location) == HYPRE_EXEC_DEVICE)
   {
      valid = 0;
   }
#endif
   if (valid)
   {
      valid = hypre_SStructMatrixBoxSegments(A, grid, boxes, -1, &segments, &count, &nnz);
   }

   if (U)
   {
      diag_nnz = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(U));
      offd_nnz = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(U));
      if (diag_nnz > HYPRE_INT_MAX - nnz ||
          offd_nnz > HYPRE_INT_MAX - nnz - diag_nnz)
      {
         valid = 0;
      }
   }

   /* All ranks must select the same path before IJ's collective assembly */
   hypre_MPI_Allreduce(&valid, &all_valid, 1, hypre_MPI_INT, hypre_MPI_MIN, comm);
   if (!all_valid)
   {
      hypre_TFree(segments, HYPRE_MEMORY_HOST);

      return 0;
   }

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   if (hypre_GetExecPolicy1(location) == HYPRE_EXEC_DEVICE)
   {
      HYPRE_IJMatrixPartialClone(ij_A, &ij);
      hypre_SStructMatrixCopyBoxesDevice(ij, segments, count, nnz, U);
      *result = ij;
      hypre_TFree(segments, HYPRE_MEMORY_HOST);

      return 1;
   }
#endif

#if defined(HYPRE_USING_GPU)
   if (hypre_GetExecPolicy1(location) == HYPRE_EXEC_HOST)
#endif
   {
      HYPRE_IJMatrixPartialClone(ij_A, &ij);
      hypre_SStructMatrixCopyBoxesHost(ij, segments, count, nnz, U);
      *result = ij;
      hypre_TFree(segments, HYPRE_MEMORY_HOST);

      return 1;
   }

   hypre_TFree(segments, HYPRE_MEMORY_HOST);

   return 0;
}

static HYPRE_Int
hypre_SStructMatrixBoxesToUMatrixGeneral( hypre_SStructMatrix   *A,
                                          hypre_SStructGrid     *grid,
                                          hypre_IJMatrix       **ij_Ahat_ptr,
                                          hypre_BoxArray      ***convert_boxa)
{
   HYPRE_Int              ndim            = hypre_SStructMatrixNDim(A);
   HYPRE_Int              nparts          = hypre_SStructMatrixNParts(A);
   HYPRE_Int             *Sentries        = hypre_SStructMatrixSEntries(A);
   HYPRE_IJMatrix         ij_A            = hypre_SStructMatrixIJMatrix(A);
   HYPRE_MemoryLocation   memory_location = hypre_IJMatrixMemoryLocation(ij_A);

   hypre_SStructPGrid    *pgrid;
   hypre_StructGrid      *sgrid;
   hypre_SStructStencil  *stencil;

   hypre_SStructPMatrix  *pA;
   hypre_IJMatrix        *ij_Ahat;
   hypre_AuxParCSRMatrix *aux_matrix;

   HYPRE_Int             *split;
   hypre_BoxArray        *grid_boxes;
   hypre_Box             *box       = hypre_BoxCreate(ndim);
   hypre_Box             *ghost_box = hypre_BoxCreate(ndim);
   hypre_Box             *grid_box;
   hypre_Box             *convert_box;

   hypre_Index            stride;
   HYPRE_Int              nrows;
   HYPRE_Int              ncols;
   HYPRE_Int             *row_sizes = NULL;
   HYPRE_Complex         *values;

   HYPRE_BigInt           sizes[4];
   HYPRE_Int              entry, part, var, nvars;
   HYPRE_Int              nnzrow;
   HYPRE_Int              nvalues, i, j;
   HYPRE_Int             *num_ghost;
   HYPRE_Int              nSentries;
   HYPRE_Int              m = 0;
   hypre_Index            loop_size;
   hypre_IndexRef         start;
#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy  exec  = hypre_GetExecPolicy1(memory_location);
#endif

#if defined(HYPRE_DEBUG) && defined(DEBUG_MATCONV)
   char                   msg[512];
#endif

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("SStructMatrixBoxesToUMatrix");

   /* Get row and column ranges */
   HYPRE_IJMatrixGetLocalRange(ij_A, &sizes[0], &sizes[1], &sizes[2], &sizes[3]);
   nrows = (HYPRE_Int) (sizes[1] - sizes[0] + 1);
   ncols = (HYPRE_Int) (sizes[3] - sizes[2] + 1);

   /* Set row sizes */
   HYPRE_ANNOTATE_REGION_BEGIN("%s", "Set rowsizes");
   nvalues = 0;
   hypre_SetIndex(stride, 1);
#if defined(HYPRE_USING_GPU)
   if (exec == HYPRE_EXEC_HOST)
#endif
   {
      if (!*ij_Ahat_ptr)
      {
         row_sizes = hypre_CTAlloc(HYPRE_Int, nrows, HYPRE_MEMORY_HOST);
      }
   }

   for (part = 0; part < nparts; part++)
   {
      pA    = hypre_SStructMatrixPMatrix(A, part);
      pgrid = hypre_SStructGridPGrid(grid, part);
      nvars = hypre_SStructPMatrixNVars(pA);

      for (var = 0; var < nvars; var++)
      {
         sgrid   = hypre_SStructPGridSGrid(pgrid, var);
         split   = hypre_SStructMatrixSplit(A, part, var);
         stencil = hypre_SStructPMatrixStencil(pA, var);
         grid_boxes = hypre_StructGridBoxes(sgrid);
         num_ghost  = hypre_StructGridNumGhost(sgrid);

         nnzrow = hypre_SStructStencilSize(stencil);

         hypre_ForBoxI(i, grid_boxes)
         {
            grid_box = hypre_BoxArrayBox(grid_boxes, i);
            hypre_CopyBox(grid_box, ghost_box);
            hypre_BoxGrowByArray(ghost_box, num_ghost);

            hypre_ForBoxI(j, convert_boxa[part][var])
            {
               convert_box = hypre_BoxArrayBox(convert_boxa[part][var], j);

#if defined(HYPRE_USING_GPU)
               if (exec == HYPRE_EXEC_HOST)
#endif
               {
                  if (!*ij_Ahat_ptr)
                  {
                     /* WM: do I need to check whether there is a non-trivial
                            intersection, or is that handled automatically? */
                     hypre_IntersectBoxes(grid_box, convert_box, box);
                     start = hypre_BoxIMin(box);
                     hypre_BoxGetSize(box, loop_size);
                     hypre_BoxLoop1Begin(ndim, loop_size, ghost_box, start, stride, mi);
                     {
                        row_sizes[m + mi] = nnzrow;
                     }
                     hypre_BoxLoop1End(mi);
                  }
               }

               nvalues = hypre_max(nvalues, nnzrow * hypre_BoxVolume(convert_box));
            } /* Loop over convert_boxa[part][var] */

#if defined(HYPRE_USING_GPU)
            if (exec == HYPRE_EXEC_HOST)
#endif
            {
               m += hypre_BoxVolume(ghost_box);
            }
         } /* Loop over grid_boxes */
      } /* Loop over vars */
   } /* Loop over parts */
   hypre_BoxDestroy(box);
   hypre_BoxDestroy(ghost_box);
   HYPRE_ANNOTATE_REGION_END("%s", "Set rowsizes");

   /* Create and initialize ij_Ahat */
   if (*ij_Ahat_ptr)
   {
      ij_Ahat = *ij_Ahat_ptr;
   }
   else
   {
      HYPRE_ANNOTATE_REGION_BEGIN("%s", "Create Matrix");
      HYPRE_IJMatrixPartialClone(ij_A, &ij_Ahat);
      if (row_sizes)
      {
         hypre_AuxParCSRMatrixCreate(&aux_matrix, nrows, ncols, row_sizes);
         hypre_IJMatrixTranslator(ij_Ahat) = aux_matrix;
      }
      HYPRE_IJMatrixInitialize_v2(ij_Ahat, memory_location);
      HYPRE_ANNOTATE_REGION_END("%s", "Create Matrix");
   }

   /* Allocate memory */
   values = hypre_CTAlloc(HYPRE_Complex, nvalues, memory_location);

   /* Set entries of ij_Ahat */
   for (part = 0; part < nparts; part++)
   {
      pA    = hypre_SStructMatrixPMatrix(A, part);
      pgrid = hypre_SStructGridPGrid(grid, part);
      nvars = hypre_SStructPMatrixNVars(pA);

      for (var = 0; var < nvars; var++)
      {
         split   = hypre_SStructMatrixSplit(A, part, var);
         stencil = hypre_SStructPMatrixStencil(pA, var);

         nSentries = 0;
         for (entry = 0; entry < hypre_SStructStencilSize(stencil); entry++)
         {
            if (split[entry] > -1)
            {
               Sentries[nSentries] = split[entry];
               nSentries++;
            }
         }

         hypre_ForBoxI(i, convert_boxa[part][var])
         {
            convert_box = hypre_BoxArrayBox(convert_boxa[part][var], i);

            hypre_assert(hypre_BoxVolume(convert_box) > 0);
            hypre_assert(hypre_BoxVolume(convert_box) <= nvalues);

#if defined(HYPRE_DEBUG) && defined(DEBUG_MATCONV)
            hypre_sprintf(msg, "Part %d - Box %d - ", part, i);
            hypre_BoxPrintDebug(convert_box);
            HYPRE_ANNOTATE_REGION_BEGIN("%s %d %s %d", "Get values part", part, "convert_box", i);
#endif
            /* GET values from this box */
            HYPRE_ANNOTATE_REGION_BEGIN("%s", "Get entries");
            hypre_SStructPMatrixSetBoxValues(pA, convert_box, var,
                                             nSentries, Sentries, convert_box, values, -1);
            HYPRE_ANNOTATE_REGION_END("%s", "Get entries");

#if defined(HYPRE_DEBUG) && defined(DEBUG_MATCONV)
            HYPRE_ANNOTATE_REGION_END("%s %d %s %d", "Get values part", part, "convert_box", i);
            HYPRE_ANNOTATE_REGION_BEGIN("%s %d %s %d", "Set values part", part, "convert_box", i);
#endif
            /* SET values to ij_Ahat */
            HYPRE_ANNOTATE_REGION_BEGIN("%s", "Set entries");
            hypre_SStructUMatrixSetBoxValuesHelper(A, part, convert_box,
                                                   var, nSentries, Sentries,
                                                   convert_box, values, 0, ij_Ahat);
            HYPRE_ANNOTATE_REGION_END("%s", "Set entries");
#if defined(HYPRE_DEBUG) && defined(DEBUG_MATCONV)
            HYPRE_ANNOTATE_REGION_END("%s %d %s %d", "Set values part", part, "convert_box", i);
#endif
         } /* Loop over convert_boxa */
      } /* Loop over vars */
   } /* Loop over parts */

   /* Free memory */
   hypre_TFree(values, memory_location);

   /* Assemble ij_A */
   HYPRE_IJMatrixAssemble(ij_Ahat);

   /* Set pointer to ij_Ahat */
   *ij_Ahat_ptr = ij_Ahat;

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

HYPRE_Int
hypre_SStructMatrixBoxesToUMatrix( hypre_SStructMatrix   *A,
                                   hypre_SStructGrid     *grid,
                                   hypre_IJMatrix       **result,
                                   hypre_BoxArray      ***boxes )
{
   if (!hypre_SStructMatrixCopyBoxes(A, grid, result, boxes, NULL))
   {
      hypre_SStructMatrixBoxesToUMatrixGeneral(A, grid, result, boxes);
   }

   return hypre_error_flag;
}

HYPRE_Int
hypre_SStructMatrixFreeSupport( hypre_SStructGrid  *grid,
                                hypre_BoxArray   ***boxes )
{
   HYPRE_Int  nparts = hypre_SStructGridNParts(grid);
   HYPRE_Int  nvars;
   HYPRE_Int  part, var;

   for (part = 0; part < nparts; part++)
   {
      nvars = hypre_SStructPGridNVars(hypre_SStructGridPGrid(grid, part));
      for (var = 0; var < nvars; var++)
      {
         hypre_BoxArrayDestroy(boxes[part][var]);
      }
      hypre_TFree(boxes[part], HYPRE_MEMORY_HOST);
   }
   hypre_TFree(boxes, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}



#if defined(HYPRE_USING_GPU)
/*--------------------------------------------------------------------------
 * Marker of the rows that have entries in the diagonal or off-diagonal block
 *--------------------------------------------------------------------------*/

struct hypre_SStructRowHasEntries
{
   HYPRE_Int  *di;
   HYPRE_Int  *oi;

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   __host__ __device__
#endif
   HYPRE_Int operator()(HYPRE_Int row) const
   {
      return (di[row + 1] > di[row]) || (oi && oi[row + 1] > oi[row]);
   }
};
#endif

/*--------------------------------------------------------------------------
 * Returns the boxes of the rows of B with entries (and also the rows given by
 * its columns when include_columns is set). If row_support is not NULL, the
 * boxes of the rows only are also returned in it.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatrixCollectSupport( hypre_SStructGrid   *grid,
                                   hypre_ParCSRMatrix  *B,
                                   HYPRE_Int            include_columns,
                                   hypre_BoxArray    ****boxes_ptr,
                                   hypre_BoxArray    ****row_support )
{
   hypre_CSRMatrix       *diag     = hypre_ParCSRMatrixDiag(B);
   hypre_CSRMatrix       *offd     = hypre_ParCSRMatrixOffd(B);
   HYPRE_Int             *di       = hypre_CSRMatrixI(diag);
   HYPRE_Int             *oi       = hypre_CSRMatrixI(offd);
   HYPRE_Int              nrows    = hypre_CSRMatrixNumRows(diag);
   HYPRE_MemoryLocation   location = hypre_ParCSRMatrixMemoryLocation(B);

   hypre_BoxArray      ***boxes;
   HYPRE_Int             *marker;
   HYPRE_Int              i;

   marker = hypre_TAlloc(HYPRE_Int, nrows, location);

#if defined(HYPRE_USING_GPU)
   if (hypre_GetExecPolicy1(location) == HYPRE_EXEC_DEVICE)
   {
      hypre_SStructRowHasEntries  has_entries = {di, oi};

#if defined(HYPRE_USING_SYCL)
      HYPRE_ONEDPL_CALL( std::transform,
                         oneapi::dpl::counting_iterator<HYPRE_Int>(0),
                         oneapi::dpl::counting_iterator<HYPRE_Int>(nrows),
                         marker,
                         has_entries );
#else
      HYPRE_THRUST_CALL( transform,
                         thrust::make_counting_iterator(0),
                         thrust::make_counting_iterator(nrows),
                         marker,
                         has_entries );
#endif
   }
   else
#endif
   {
      for (i = 0; i < nrows; i++)
      {
         marker[i] = (di && di[i + 1] > di[i]) || (oi && oi[i + 1] > oi[i]);
      }
   }

   if (row_support)
   {
      hypre_SStructGridMarkerBoxes(grid, marker, nrows, location, row_support);
   }
   if (include_columns)
   {
      hypre_ParCSRMatrixMarkColumns(B, marker);
   }
   hypre_SStructGridMarkerBoxes(grid, marker, nrows, location, &boxes);

   hypre_TFree(marker, location);

   *boxes_ptr = boxes;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Moves the entry of U at element k of segment s into its coefficient in S
 * (adding to it when action > 0) and leaves a zero in U. For action < 0, only
 * clears the coefficient in S.
 *--------------------------------------------------------------------------*/

static inline
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
__host__ __device__
#endif
void
hypre_SStructCompressEntry( const hypre_SStructMatrixCopySegment *s,
                            HYPRE_Int                             k,
                            HYPRE_BigInt                          first_row,
                            HYPRE_BigInt                          first_col,
                            HYPRE_BigInt                          last_col,
                            HYPRE_Int                            *di,
                            HYPRE_Int                            *dj,
                            HYPRE_Complex                        *da,
                            HYPRE_Int                            *oi,
                            HYPRE_Int                            *oj,
                            HYPRE_Complex                        *oa,
                            HYPRE_BigInt                         *col_map,
                            HYPRE_Int                             num_cols_offd,
                            HYPRE_Int                             action )
{
   HYPRE_Int       row, didx;
   HYPRE_BigInt    col;
   HYPRE_Complex  *u, *coef;

   hypre_SStructSegmentEntry(s, k, first_row, &row, &col, &didx);

   coef = s->data + didx;
   if (action < 0)
   {
      *coef = 0.0;
      return;
   }

   u = hypre_SStructUMatrixFindEntry(row, col, first_col, last_col, di, dj, da, oi, oj, oa,
                                     col_map, num_cols_offd);
   if (u)
   {
      *coef = action ? *coef + *u : *u;
      *u = 0.0;
   }
}

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
/*--------------------------------------------------------------------------
 * Compresses the entries of a segment. Different entries of a segment touch
 * different coefficients of S and entries of U.
 *--------------------------------------------------------------------------*/

struct hypre_SStructCompressSegment
{
   hypre_SStructMatrixCopySegment  s;
   HYPRE_BigInt                    first_row;
   HYPRE_BigInt                    first_col;
   HYPRE_BigInt                    last_col;
   HYPRE_Int                      *di;
   HYPRE_Int                      *dj;
   HYPRE_Complex                  *da;
   HYPRE_Int                      *oi;
   HYPRE_Int                      *oj;
   HYPRE_Complex                  *oa;
   HYPRE_BigInt                   *col_map;
   HYPRE_Int                       num_cols_offd;
   HYPRE_Int                       action;

   __host__ __device__
   void operator()(HYPRE_Int k) const
   {
      hypre_SStructCompressEntry(&s, k, first_row, first_col, last_col, di, dj, da, oi, oj, oa,
                                 col_map, num_cols_offd, action);
   }
};
#endif

/*--------------------------------------------------------------------------
 * Adds to (action > 0) or sets (action = 0) the coefficients of S with the
 * entries of U at the same stencil positions within a part, on host or device,
 * and zeros these entries in U. Couplings to neighbor parts stay in U and their
 * coefficients in S are cleared. Missing U entries retain their S values.
 * Supports square matrices with unit strides and one variable per part;
 * returns zero (locally) otherwise.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatrixCompressUToSBox( hypre_SStructMatrix *A,
                                    HYPRE_Int            action )
{
   hypre_SStructGraph              *graph         = hypre_SStructMatrixGraph(A);
   hypre_SStructGrid               *grid          = hypre_SStructGraphGrid(graph);
   hypre_ParCSRMatrix              *U             = hypre_SStructMatrixParCSRMatrix(A);
   hypre_CSRMatrix                 *ud            = hypre_ParCSRMatrixDiag(U);
   hypre_CSRMatrix                 *uo            = hypre_ParCSRMatrixOffd(U);
   HYPRE_Int                       *di            = hypre_CSRMatrixI(ud);
   HYPRE_Int                       *dj            = hypre_CSRMatrixJ(ud);
   HYPRE_Complex                   *da            = hypre_CSRMatrixData(ud);
   HYPRE_Int                       *oi            = hypre_CSRMatrixI(uo);
   HYPRE_Int                       *oj            = hypre_CSRMatrixJ(uo);
   HYPRE_Complex                   *oa            = hypre_CSRMatrixData(uo);
   HYPRE_BigInt                    *col_map       = hypre_ParCSRMatrixColMapOffd(U);
   HYPRE_Int                        num_cols_offd = hypre_CSRMatrixNumCols(uo);
   HYPRE_BigInt                     first_row     = hypre_ParCSRMatrixFirstRowIndex(U);
   HYPRE_BigInt                     first_col     = hypre_ParCSRMatrixFirstColDiag(U);
   HYPRE_BigInt                     last_col      = hypre_ParCSRMatrixLastColDiag(U);
#if defined(HYPRE_USING_GPU)
   HYPRE_MemoryLocation             location      = hypre_ParCSRMatrixMemoryLocation(U);
#endif

   hypre_SStructMatrixCopySegment  *segments[2]   = {NULL, NULL};
   hypre_BoxArray                ***boxes;
   HYPRE_Int                        count[2], nnz[2];
   HYPRE_Int                        valid, m, e, i, m_action;

   if (!hypre_CSRMatrixNumNonzeros(uo))
   {
      num_cols_offd = 0;
   }

   if (!hypre_SStructMatrixCopyLayout(A, grid, 1))
   {
      return 0;
   }
#if defined(HYPRE_USING_GPU) && !defined(HYPRE_USING_CUDA) && !defined(HYPRE_USING_HIP)
   if (hypre_GetExecPolicy1(location) == HYPRE_EXEC_DEVICE)
   {
      return 0;
   }
#endif

   /* Segments of the rows of U with entries: couplings within a part (moved
      from U to S) and couplings to neighbor parts (cleared in S). Both share
      the indexing, periodic-image filtering and overflow checks. */
   hypre_SStructMatrixCollectSupport(grid, U, 0, &boxes, NULL);
   valid = hypre_SStructMatrixBoxSegments(A, grid, boxes, 0, &segments[0], &count[0], &nnz[0]);
   valid = hypre_SStructMatrixBoxSegments(A, grid, boxes, 1, &segments[1], &count[1], &nnz[1]) &&
           valid;
   hypre_SStructMatrixFreeSupport(grid, boxes);
   if (!valid)
   {
      hypre_TFree(segments[0], HYPRE_MEMORY_HOST);
      hypre_TFree(segments[1], HYPRE_MEMORY_HOST);

      return 0;
   }

   for (m = 0; m < 2; m++)
   {
      m_action = m ? -1 : action;

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
      if (hypre_GetExecPolicy1(location) == HYPRE_EXEC_DEVICE)
      {
         if (num_cols_offd)
         {
            hypre_ParCSRMatrixCopyColMapOffdToDevice(U);
            col_map = hypre_ParCSRMatrixDeviceColMapOffd(U);
         }

         for (e = 0; e < count[m]; e++)
         {
            hypre_SStructCompressSegment  compress = {segments[m][e], first_row, first_col,
                                                      last_col, di, dj, da, oi, oj, oa,
                                                      col_map, num_cols_offd, m_action
                                                     };

            HYPRE_THRUST_CALL( for_each,
                               thrust::make_counting_iterator(0),
                               thrust::make_counting_iterator(segments[m][e].volume),
                               compress );
         }
      }
      else
#endif
      {
         for (e = 0; e < count[m]; e++)
         {
            for (i = 0; i < segments[m][e].volume; i++)
            {
               hypre_SStructCompressEntry(segments[m] + e, i, first_row, first_col, last_col,
                                          di, dj, da, oi, oj, oa, col_map, num_cols_offd,
                                          m_action);
            }
         }
      }
      hypre_TFree(segments[m], HYPRE_MEMORY_HOST);
   }

   return 1;
}

/*--------------------------------------------------------------------------
 * Converts the structured component of A on the rows in boxes and adds the
 * unstructured component of A (U) to it.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatrixBoxesWithU( hypre_SStructMatrix  *A,
                               hypre_BoxArray     ***boxes,
                               hypre_IJMatrix      **result )
{
   hypre_SStructGraph  *graph = hypre_SStructMatrixGraph(A);
   hypre_SStructGrid   *grid  = hypre_SStructGraphGrid(graph);
   hypre_ParCSRMatrix  *U     = hypre_SStructMatrixParCSRMatrix(A);
   hypre_ParCSRMatrix  *structured, *full;

   if (!hypre_SStructMatrixCopyBoxes(A, grid, result, boxes, U))
   {
      hypre_SStructMatrixBoxesToUMatrixGeneral(A, grid, result, boxes);
      HYPRE_IJMatrixGetObject(*result, (void **) &structured);
      hypre_ParCSRMatrixAdd(1.0, structured, 1.0, U, &full);
      hypre_ParCSRMatrixDestroy(structured);
      hypre_IJMatrixObject(*result) = full;
   }

   return hypre_error_flag;
}
