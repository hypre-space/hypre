/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_sstruct_ls.h"
#include "ssamg.h"

/*--------------------------------------------------------------------------
 * hypre_SSAMGPrintStats
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGPrintStats( void *ssamg_vdata )
{
   hypre_SSAMGData        *ssamg_data    = (hypre_SSAMGData *) ssamg_vdata;
   MPI_Comm                comm          = hypre_SSAMGDataComm(ssamg_data);
   HYPRE_Int               num_levels    = hypre_SSAMGDataNumLevels(ssamg_data);
   HYPRE_Int               non_galerkin  = hypre_SSAMGDataNonGalerkin(ssamg_data);
   HYPRE_Int               print_level   = hypre_SSAMGDataPrintLevel(ssamg_data);
   HYPRE_Int               relax_type    = hypre_SSAMGDataRelaxType(ssamg_data);
   HYPRE_Int               skip_relax    = hypre_SSAMGDataSkipRelax(ssamg_data);
   HYPRE_Int               num_pre_relax = hypre_SSAMGDataNumPreRelax(ssamg_data);
   HYPRE_Int               num_pos_relax = hypre_SSAMGDataNumPosRelax(ssamg_data);
   HYPRE_Int               num_crelax    = hypre_SSAMGDataNumCoarseRelax(ssamg_data);
   HYPRE_Int               nparts        = hypre_SSAMGDataNParts(ssamg_data);
   HYPRE_Int             **cdir_l        = hypre_SSAMGDataCdir(ssamg_data);
   HYPRE_Int             **active_l      = hypre_SSAMGDataActivel(ssamg_data);
   HYPRE_Real            **weights       = hypre_SSAMGDataRelaxWeights(ssamg_data);

   hypre_SStructMatrix   **A_l = hypre_SSAMGDataAl(ssamg_data);
   hypre_SStructPMatrix   *pmatrix;
   hypre_StructMatrix     *smatrix;
   hypre_ParCSRMatrix     *umatrix;
   hypre_StructGrid       *sgrid;
   hypre_StructStencil    *stencil;
   hypre_CSRMatrix        *diag;
   hypre_CSRMatrix        *offd;
   HYPRE_Int              *diag_i;
   HYPRE_Int              *offd_i;
   HYPRE_Complex          *diag_a;
   HYPRE_Complex          *offd_a;

   HYPRE_Int              *rownnz;
   HYPRE_Int              *diag_rownnz;
   HYPRE_Int              *offd_rownnz;
   HYPRE_Int               num_rownnz;
   HYPRE_Int               diag_num_rownnz;
   HYPRE_Int               offd_num_rownnz;

   HYPRE_Int              *global_num_rows;
   HYPRE_Int              *global_num_rownnz;
   HYPRE_Int              *global_num_nonzeros;
   HYPRE_Int              *global_min_entries;
   HYPRE_Int              *global_max_entries;
   HYPRE_Real             *global_avg_entries;
   HYPRE_Complex          *global_min_rowsum;
   HYPRE_Complex          *global_max_rowsum;
   HYPRE_Int              *global_num_parts;
   HYPRE_Int              *global_num_boxes;
   HYPRE_Int              *global_num_dofs;
   HYPRE_Int              *global_num_ghrows;
   HYPRE_Int              *global_min_stsize;
   HYPRE_Int              *global_max_stsize;
   HYPRE_Real             *global_avg_stsize;

   HYPRE_Int               stencil_size;
   HYPRE_Int               min_entries;
   HYPRE_Int               max_entries;
   HYPRE_Int               min_stsize;
   HYPRE_Int               max_stsize;
   HYPRE_Real              avg_stsize;
   HYPRE_Complex           min_rowsum;
   HYPRE_Complex           max_rowsum;
   HYPRE_Complex           rowsum;

   HYPRE_Int               num_ghrows;
   HYPRE_Int               num_boxes;
   HYPRE_Int               num_parts;
   HYPRE_Int               num_boxes_part;
   HYPRE_Int               num_dofs;
   HYPRE_Int               num_dofs_grid;

   HYPRE_Int               myid, i, ii, j, l;
   HYPRE_Int               part, vi, vj, nvars;
   HYPRE_Int               entries;
   HYPRE_Int               chunk, chunk_size, chunk_last;
   HYPRE_Int               nparts_per_line = 8;
   HYPRE_Int               offset = 2;
   HYPRE_Int               ndigits;
   HYPRE_Int               ndigits_S[6] = {7, 7, 6, 5, 5, 5};
   HYPRE_Int               ndigits_U[7] = {6, 6, 6, 9, 5, 5, 5};
   HYPRE_Int               header[5];
   HYPRE_Real              send_buffer[5];
   HYPRE_Real              recv_buffer[5];

   hypre_MPI_Comm_rank(comm, &myid);

   if (print_level == 0)
   {
      return hypre_error_flag;
   }

   /* Allocate memory */
   if (myid == 0)
   {
      global_num_rows     = hypre_CTAlloc(HYPRE_Int, num_levels);
      global_num_rownnz   = hypre_CTAlloc(HYPRE_Int, num_levels);
      global_num_nonzeros = hypre_CTAlloc(HYPRE_Int, num_levels);
      global_min_entries  = hypre_CTAlloc(HYPRE_Int, num_levels);
      global_max_entries  = hypre_CTAlloc(HYPRE_Int, num_levels);
      global_avg_entries  = hypre_CTAlloc(HYPRE_Real, num_levels);
      global_min_rowsum   = hypre_CTAlloc(HYPRE_Complex, num_levels);
      global_max_rowsum   = hypre_CTAlloc(HYPRE_Complex, num_levels);
      global_num_parts    = hypre_CTAlloc(HYPRE_Int, num_levels);
      global_num_boxes    = hypre_CTAlloc(HYPRE_Int, num_levels);
      global_num_dofs     = hypre_CTAlloc(HYPRE_Int, num_levels);
      global_num_ghrows   = hypre_CTAlloc(HYPRE_Int, num_levels);
      global_min_stsize   = hypre_CTAlloc(HYPRE_Int, num_levels);
      global_max_stsize   = hypre_CTAlloc(HYPRE_Int, num_levels);
      global_avg_stsize   = hypre_CTAlloc(HYPRE_Real, num_levels);
   }

   /* Gather UMatrix info */
   for (l = 0; l < num_levels; l++)
   {
      umatrix = hypre_SStructMatrixParCSRMatrix(A_l[l]);
      hypre_ParCSRMatrixSetNumRownnz(umatrix);
      if (!hypre_ParCSRMatrixNumNonzeros(umatrix))
      {
         hypre_ParCSRMatrixSetNumNonzeros(umatrix);
      }

      diag   = hypre_ParCSRMatrixDiag(umatrix);
      offd   = hypre_ParCSRMatrixOffd(umatrix);
      diag_i = hypre_CSRMatrixI(diag);
      offd_i = hypre_CSRMatrixI(offd);
      diag_a = hypre_CSRMatrixData(diag);
      offd_a = hypre_CSRMatrixData(offd);
      diag_rownnz = hypre_CSRMatrixRownnz(diag);
      offd_rownnz = hypre_CSRMatrixRownnz(offd);
      diag_num_rownnz = hypre_CSRMatrixNumRownnz(diag);
      offd_num_rownnz = hypre_CSRMatrixNumRownnz(offd);
      hypre_MergeOrderedArrays(diag_num_rownnz, diag_rownnz,
                               offd_num_rownnz, offd_rownnz,
                               &num_rownnz, &rownnz);

      if (myid == 0)
      {
         global_num_rows[l]     = hypre_ParCSRMatrixGlobalNumRows(umatrix);
         global_num_rownnz[l]   = hypre_ParCSRMatrixGlobalNumRownnz(umatrix);
         global_num_nonzeros[l] = hypre_ParCSRMatrixNumNonzeros(umatrix);
         if (global_num_rownnz[l])
         {
            global_avg_entries[l] = global_num_nonzeros[l] / (HYPRE_Real) global_num_rownnz[l];
         }
      }

      if (num_rownnz)
      {
         min_entries = HYPRE_INT_MAX;
         max_entries = HYPRE_INT_MIN;
         min_rowsum  = HYPRE_REAL_MAX;
         max_rowsum  = - min_rowsum;
      }
      else
      {
         min_entries = 0;
         max_entries = 0;
         min_rowsum  = 0.0;
         max_rowsum  = 0.0;
      }

      for (i = 0; i < num_rownnz; i++)
      {
         ii = rownnz[i];

         entries = (diag_i[ii+1] - diag_i[ii]) + (offd_i[ii+1] - offd_i[ii]);
         min_entries = hypre_min(entries, min_entries);
         max_entries = hypre_max(entries, max_entries);

         rowsum = 0.0;
         for (j = diag_i[ii]; j < diag_i[ii+1]; j++)
         {
            rowsum += diag_a[j];
         }
         for (j = offd_i[ii]; j < offd_i[ii+1]; j++)
         {
            rowsum += offd_a[j];
         }
         min_rowsum = hypre_min(rowsum, min_rowsum);
         max_rowsum = hypre_max(rowsum, max_rowsum);
      }

      send_buffer[0] = - (HYPRE_Real) min_entries;
      send_buffer[1] =   (HYPRE_Real) max_entries;
      send_buffer[2] = - min_rowsum;
      send_buffer[3] =   max_rowsum;

      hypre_MPI_Reduce(send_buffer, recv_buffer, 4, HYPRE_MPI_REAL, hypre_MPI_MAX, 0, comm);

      if (myid == 0)
      {
         global_min_entries[l] = - (HYPRE_Int) recv_buffer[0];
         global_max_entries[l] =   (HYPRE_Int) recv_buffer[1];
         global_min_rowsum[l]  = - recv_buffer[2];
         global_max_rowsum[l]  =   recv_buffer[3];
      }

      hypre_TFree(rownnz);
   }

   /* Gather SMatrix info */
   for (l = 0; l < num_levels; l++)
   {
      nparts = hypre_SStructMatrixNParts(A_l[l]);
      num_ghrows = hypre_SStructMatrixRanGhlocalSize(A_l[l]);

      min_stsize = HYPRE_INT_MAX;
      max_stsize = HYPRE_INT_MIN;

      num_dofs = num_boxes = num_parts = avg_stsize = 0;
      for (part = 0; part < nparts; part++)
      {
         pmatrix = hypre_SStructMatrixPMatrix(A_l[l], part);
         nvars   = hypre_SStructPMatrixNVars(pmatrix);

         num_boxes_part = 0;
         for (vi = 0; vi < nvars; vi++)
         {
            for (vj = 0; vj < nvars; vj++)
            {
               smatrix = hypre_SStructPMatrixSMatrix(pmatrix, vi, vj);
               if (smatrix)
               {
                  sgrid = hypre_StructMatrixGrid(smatrix);
                  num_dofs_grid   = hypre_StructGridLocalSize(sgrid);
                  num_boxes_part += hypre_StructGridNumBoxes(sgrid);
                  num_dofs   += num_dofs_grid;
                  num_ghrows -= num_dofs_grid;

                  if (num_dofs_grid)
                  {
                     stencil = hypre_StructMatrixStencil(smatrix);
                     stencil_size = hypre_StructStencilSize(stencil);
                     min_stsize = hypre_min(stencil_size, min_stsize);
                     max_stsize = hypre_max(stencil_size, max_stsize);
                     avg_stsize += stencil_size*num_dofs_grid;
                  }
               }
            }
         }

         num_boxes += num_boxes_part;
         if (num_boxes_part)
         {
            num_parts++;
         }
      }

      send_buffer[0] = (HYPRE_Real) num_parts;
      send_buffer[1] = (HYPRE_Real) num_boxes;
      send_buffer[2] = (HYPRE_Real) num_dofs;
      send_buffer[3] = (HYPRE_Real) num_ghrows;
      send_buffer[4] = (HYPRE_Real) avg_stsize;

      hypre_MPI_Reduce(send_buffer, recv_buffer, 5, HYPRE_MPI_REAL, hypre_MPI_SUM, 0, comm);

      if (myid == 0)
      {
         global_num_parts[l]  = (HYPRE_Int) recv_buffer[0];
         global_num_boxes[l]  = (HYPRE_Int) recv_buffer[1];
         global_num_dofs[l]   = (HYPRE_Int) recv_buffer[2];
         global_num_ghrows[l] = (HYPRE_Int) recv_buffer[3];
         global_avg_stsize[l] = recv_buffer[4] / global_num_dofs[l];
      }

      send_buffer[0] = - (HYPRE_Real) min_stsize;
      send_buffer[1] =   (HYPRE_Real) max_stsize;

      hypre_MPI_Reduce(send_buffer, recv_buffer, 2, HYPRE_MPI_REAL, hypre_MPI_MAX, 0, comm);

      if (myid == 0)
      {
         global_min_stsize[l] = - (HYPRE_Int) recv_buffer[0];
         global_max_stsize[l] =   (HYPRE_Int) recv_buffer[1];
      }
   }

   /* Print statistics */
   if (myid == 0)
   {
      hypre_printf("\nSSAMG Setup Parameters:\n\n");

      if (print_level > -1)
      {
         /* Print coarsening direction */
         hypre_printf("Coarsening direction:\n\n");
         chunk_size = hypre_min(nparts, nparts_per_line);
         for (chunk = 0; chunk < nparts; chunk += chunk_size)
         {
            ndigits = 4;
            hypre_printf("lev   ");
            chunk_last = hypre_min(chunk + chunk_size, nparts);
            for (part = chunk; part < chunk_last; part++)
            {
               hypre_printf("pt. %d  ", part);
               ndigits += 7;
            }
            hypre_printf("\n");
            for (i = 0; i < ndigits; i++) hypre_printf("%s", "=");
            hypre_printf("\n");
            for (l = 0; l < (num_levels - 1); l++)
            {
               hypre_printf("%3d  ", l);
               for (part = chunk; part < chunk_last; part++)
               {
                  hypre_printf("%6d ", cdir_l[l][part]);
               }
               hypre_printf("\n");
            }
            hypre_printf("\n\n");
         }

         /* Print active parts */
         if (skip_relax > 0)
         {
            hypre_printf("Active parts:\n\n");
            chunk_size = hypre_min(nparts, nparts_per_line);
            for (chunk = 0; chunk < nparts; chunk += chunk_size)
            {
               ndigits = 4;
               hypre_printf("lev   ");
               chunk_last = hypre_min(chunk + chunk_size, nparts);
               for (part = chunk; part < chunk_last; part++)
               {
                  hypre_printf("pt. %d  ", part);
                  ndigits += 7;
               }
               hypre_printf("\n");
               for (i = 0; i < ndigits; i++) hypre_printf("%s", "=");
               hypre_printf("\n");
               for (l = 0; l < num_levels; l++)
               {
                  hypre_printf("%3d  ", l);
                  for (part = chunk; part < chunk_last; part++)
                  {
                     hypre_printf("%6d ", active_l[l][part]);
                  }
                  hypre_printf("\n");
               }
               hypre_printf("\n\n");
            }
         }

         /* Print relaxation weights */
         if (relax_type > 0)
         {
            hypre_printf("Relaxation weights:\n\n");
            chunk_size = hypre_min(nparts, nparts_per_line);
            for (chunk = 0; chunk < nparts; chunk += chunk_size)
            {
               ndigits = 4;
               hypre_printf("lev   ");
               chunk_last = hypre_min(chunk + chunk_size, nparts);
               for (part = chunk; part < chunk_last; part++)
               {
                  hypre_printf("pt. %d  ", part);
                  ndigits += 7;
               }
               hypre_printf("\n");
               for (i = 0; i < ndigits; i++) hypre_printf("%s", "=");
               hypre_printf("\n");
               for (l = 0; l < num_levels; l++)
               {
                  hypre_printf("%3d  ", l);
                  for (part = chunk; part < chunk_last; part++)
                  {
                     hypre_printf("%6.2f ", weights[l][part]);
                  }
                  hypre_printf("\n");
               }
               hypre_printf("\n\n");
            }
         }
      }

      /* Print SMatrix info */
      for (l = 0; l < num_levels; l++)
      {
         ndigits_S[0] = hypre_max(hypre_ndigits(global_num_parts[l]) + offset, ndigits_S[0]);
         ndigits_S[1] = hypre_max(hypre_ndigits(global_num_boxes[l]) + offset, ndigits_S[1]);
         ndigits_S[2] = hypre_max(hypre_ndigits(global_num_dofs[l]) + offset, ndigits_S[2]);
         ndigits_S[3] = hypre_max(hypre_ndigits(global_min_stsize[l]) + offset, ndigits_S[3]);
         ndigits_S[4] = hypre_max(hypre_ndigits(global_max_stsize[l]) + offset, ndigits_S[4]);
         ndigits_S[5] = hypre_max(hypre_ndigits(global_avg_stsize[l]) + offset, ndigits_S[5]);
      }

      header[0]  = 3 + (ndigits_S[0] + ndigits_S[1])/2;
      header[1]  = ndigits_S[2] + ndigits_S[3] + ndigits_S[4] + ndigits_S[5] +
                   (3 + ndigits_S[0] + ndigits_S[1] - header[0]);
      header[2]  = header[0] + header[1];
      //header[2]  = header[0] + header[1] + 22;

      /* Print first line of header */
      hypre_printf("SMatrix info:\n\n");
      hypre_printf("%*s", header[0], "active");
      hypre_printf("%*s", header[1], "stencil size");
      //hypre_printf("%22s", "row sums");
      hypre_printf("\n");

      /* Print second line of header */
      hypre_printf("%s", "lev");
      hypre_printf("%*s", ndigits_S[0], "parts");
      hypre_printf("%*s", ndigits_S[1], "boxes");
      hypre_printf("%*s", ndigits_S[2], "DOFs");
      hypre_printf("%*s", ndigits_S[3], "min");
      hypre_printf("%*s", ndigits_S[4], "max");
      hypre_printf("%*s", ndigits_S[5], "avg");
      //hypre_printf("%11s %10s", "min", "max");
      hypre_printf("\n");

      /* Print third line of header */
      for (i = 0; i < header[2]; i++)
      {
         hypre_printf("%s", "=");
      }
      hypre_printf("\n");

      /* Print info */
      for (l = 0; l < num_levels; l++)
      {
         hypre_printf("%3d", l);
         hypre_printf("%*d", ndigits_S[0], global_num_parts[l]);
         hypre_printf("%*d", ndigits_S[1], global_num_boxes[l]);
         hypre_printf("%*d", ndigits_S[2], global_num_dofs[l]);
         hypre_printf("%*d", ndigits_S[3], global_min_stsize[l]);
         hypre_printf("%*d", ndigits_S[4], global_max_stsize[l]);
         hypre_printf("%*.1f", ndigits_S[5], global_avg_stsize[l]);
         hypre_printf("\n");
      }
      hypre_printf("\n\n");

      /* Print UMatrix info */
      for (l = 0; l < num_levels; l++)
      {
         ndigits_U[0] = hypre_max(hypre_ndigits(global_num_rows[l]) + offset, ndigits_U[0]);
         ndigits_U[1] = hypre_max(hypre_ndigits(global_num_ghrows[l]) + offset, ndigits_U[1]);
         ndigits_U[2] = hypre_max(hypre_ndigits(global_num_rownnz[l]) + offset, ndigits_U[2]);
         ndigits_U[3] = hypre_max(hypre_ndigits(global_num_nonzeros[l]) + offset, ndigits_U[3]);
         ndigits_U[4] = hypre_max(hypre_ndigits(global_min_entries[l]) + offset, ndigits_U[4]);
         ndigits_U[5] = hypre_max(hypre_ndigits(global_max_entries[l]) + offset, ndigits_U[5]);
         ndigits_U[6] = hypre_max(hypre_ndigits(global_avg_entries[l]) + offset, ndigits_U[6]);
      }

      header[0] = 3 + ndigits_U[0] + ndigits_U[1];
      header[1] = ndigits_U[2];
      header[2] = ndigits_U[3];
      header[3] = hypre_max(16, ndigits_U[4] + ndigits_U[5] + ndigits_U[6]);
      header[4] = header[0] + header[1] + header[2] + header[3] + 22;
      ndigits_U[4] = 16 - ndigits_U[5] - ndigits_U[6];

      /* Print first line of header */
      hypre_printf("UMatrix info:\n\n");
      hypre_printf("%*s", header[0], "ghost");
      hypre_printf("%*s", header[1], "nnz");
      hypre_printf("%*s", header[2], "nnz");
      hypre_printf("%*s", header[3], "entries/nnzrow");
      hypre_printf("%22s\n", "row sums");

      /* Print second line of header */
      hypre_printf("%s", "lev");
      hypre_printf("%*s", ndigits_U[0], "rows");
      hypre_printf("%*s", ndigits_U[1], "rows");
      hypre_printf("%*s", ndigits_U[2], "rows");
      hypre_printf("%*s", ndigits_U[3], "entries");
      hypre_printf("%*s", ndigits_U[4], "min");
      hypre_printf("%*s", ndigits_U[5], "max");
      hypre_printf("%*s", ndigits_U[6], "avg");
      hypre_printf("%11s %10s\n", "min", "max");

      /* Print third line of header */
      for (i = 0; i < header[4]; i++)
      {
         hypre_printf("%s", "=");
      }
      hypre_printf("\n");

      /* Print info */
      for (l = 0; l < num_levels; l++)
      {
         hypre_printf("%3d", l);
         hypre_printf("%*d", ndigits_U[0], global_num_rows[l]);
         hypre_printf("%*d", ndigits_U[1], global_num_ghrows[l]);
         hypre_printf("%*d", ndigits_U[2], global_num_rownnz[l]);
         hypre_printf("%*d", ndigits_U[3], global_num_nonzeros[l]);
         hypre_printf("%*d", ndigits_U[4], global_min_entries[l]);
         hypre_printf("%*d", ndigits_U[5], global_max_entries[l]);
         hypre_printf("%*.1f", ndigits_U[6], global_avg_entries[l]);
         hypre_printf("%11.2e", global_min_rowsum[l]);
         hypre_printf("%11.2e", global_max_rowsum[l]);
         hypre_printf("\n");
      }
      hypre_printf("\n\n");

      /* SSAMG details */
      if (non_galerkin)
      {
         hypre_printf("RAP type: non-galerkin\n");
      }
      else
      {
         hypre_printf("RAP type: galerkin\n");
      }
      hypre_printf("Relaxation skip: ");
      if (skip_relax)
      {
         hypre_printf("on\n");
      }
      else
      {
         hypre_printf("none\n");
      }
      hypre_printf("Relaxation type: ");
      if (relax_type == 0)
      {
         hypre_printf("Jacobi\n");
      }
      else if (relax_type == 1)
      {
         hypre_printf("Weighted Jacobi\n");
      }
      else if (relax_type == 2)
      {
         hypre_printf("Red-Black Gauss-Seidel\n");
      }
      hypre_printf("Number of pre-sweeps: %d\n", num_pre_relax);
      hypre_printf("Number of pos-sweeps: %d\n", num_pos_relax);
      hypre_printf("Number of coarse-sweeps: %d\n", num_crelax);
      hypre_printf("Number of levels: %d\n", num_levels);

      hypre_printf("\n\n");
   }

   if (myid == 0)
   {
      hypre_TFree(global_num_rows);
      hypre_TFree(global_num_rownnz);
      hypre_TFree(global_num_nonzeros);
      hypre_TFree(global_min_entries);
      hypre_TFree(global_max_entries);
      hypre_TFree(global_avg_entries);
      hypre_TFree(global_min_rowsum);
      hypre_TFree(global_max_rowsum);
      hypre_TFree(global_num_parts);
      hypre_TFree(global_num_boxes);
      hypre_TFree(global_num_dofs);
      hypre_TFree(global_num_ghrows);
      hypre_TFree(global_min_stsize);
      hypre_TFree(global_max_stsize);
      hypre_TFree(global_avg_stsize);
   }

   return hypre_error_flag;
}
