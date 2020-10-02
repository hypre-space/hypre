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

#include "_hypre_sstruct_ls.h"
#include "ssamg.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
hypre_SSAMGCreate( hypre_MPI_Comm comm )
{
   hypre_SSAMGData   *ssamg_data;

   ssamg_data = hypre_CTAlloc(hypre_SSAMGData, 1);

   (ssamg_data -> comm)       = comm;
   (ssamg_data -> time_index) = hypre_InitializeTiming("SSAMG");

   /* set defaults */
   (ssamg_data -> tol)              = 1.0e-06;
   (ssamg_data -> max_iter)         = 200;
   (ssamg_data -> rel_change)       = 0;
   (ssamg_data -> zero_guess)       = 0;
   (ssamg_data -> max_levels)       = 0;
   (ssamg_data -> relax_type)       = 0;
   (ssamg_data -> usr_relax_weight) = 0.0;
   (ssamg_data -> num_pre_relax)    = 1;
   (ssamg_data -> num_post_relax)   = 1;
   (ssamg_data -> num_coarse_relax) = -1;
   (ssamg_data -> logging)          = 0;
   (ssamg_data -> print_level)      = 0;

   /* initialize */
   (ssamg_data -> nparts)           = -1;
   (ssamg_data -> num_levels)       = -1;

   return (void *) ssamg_data;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGDestroy( void *ssamg_vdata )
{
   hypre_SSAMGData   *ssamg_data = (hypre_SSAMGData *) ssamg_vdata;

   HYPRE_Int          num_levels;
   HYPRE_Int          max_levels;
   HYPRE_Int          l, p;

   if (ssamg_data)
   {
      if (hypre_SSAMGDataLogging(ssamg_data) > 0)
      {
         hypre_TFree(ssamg_data -> norms);
         hypre_TFree(ssamg_data -> rel_norms);
      }

      if (hypre_SSAMGDataNumLevels(ssamg_data) > -1)
      {
         num_levels = hypre_SSAMGDataNumLevels(ssamg_data);
         max_levels = hypre_SSAMGDataMaxLevels(ssamg_data);

         hypre_SSAMGRelaxDestroy(ssamg_data -> relax_data_l[0]);
         hypre_SStructMatvecDestroy(ssamg_data -> matvec_data_l[0]);
         HYPRE_SStructVectorDestroy(ssamg_data -> b_l[0]);
         HYPRE_SStructVectorDestroy(ssamg_data -> x_l[0]);
         HYPRE_SStructVectorDestroy(ssamg_data -> tx_l[0]);
         HYPRE_SStructMatrixDestroy(ssamg_data -> A_l[0]);
         HYPRE_SStructGridDestroy(ssamg_data -> grid_l[0]);
         hypre_TFree(ssamg_data -> cdir_l[0]);
         hypre_TFree(ssamg_data -> relax_weights[0]);
         for (l = 1; l < num_levels; l++)
         {
            hypre_SSAMGRelaxDestroy(ssamg_data -> relax_data_l[l]);
            hypre_SStructMatvecDestroy(ssamg_data -> matvec_data_l[l]);
            HYPRE_SStructGridDestroy(ssamg_data -> grid_l[l]);
            HYPRE_SStructVectorDestroy(ssamg_data -> b_l[l]);
            HYPRE_SStructVectorDestroy(ssamg_data -> x_l[l]);
            HYPRE_SStructVectorDestroy(ssamg_data -> tx_l[l]);
            HYPRE_SStructMatrixDestroy(ssamg_data -> A_l[l]);
            HYPRE_SStructMatrixDestroy(ssamg_data -> P_l[l-1]);
            HYPRE_SStructMatrixDestroy(ssamg_data -> RT_l[l-1]);
            hypre_SStructMatvecDestroy(ssamg_data -> restrict_data_l[l-1]);
            hypre_SStructMatvecDestroy(ssamg_data -> interp_data_l[l-1]);
            hypre_TFree(ssamg_data -> cdir_l[l]);
            hypre_TFree(ssamg_data -> relax_weights[l]);
         }

         for (l = num_levels; l < max_levels; l++)
         {
            hypre_TFree(ssamg_data -> relax_weights[l]);
         }

         hypre_TFree(ssamg_data -> b_l);
         hypre_TFree(ssamg_data -> x_l);
         hypre_TFree(ssamg_data -> tx_l);
         hypre_TFree(ssamg_data -> A_l);
         hypre_TFree(ssamg_data -> P_l);
         hypre_TFree(ssamg_data -> RT_l);
         hypre_TFree(ssamg_data -> grid_l);
         hypre_TFree(ssamg_data -> cdir_l);
         hypre_TFree(ssamg_data -> relax_weights);
         hypre_TFree(ssamg_data -> relax_data_l);
         hypre_TFree(ssamg_data -> matvec_data_l);
         hypre_TFree(ssamg_data -> restrict_data_l);
         hypre_TFree(ssamg_data -> interp_data_l);

         ssamg_data -> e_l = NULL;
         ssamg_data -> r_l = NULL;
      }

      for (p = 0; p < hypre_SSAMGDataNParts(ssamg_data); p++)
      {
         hypre_TFree(ssamg_data -> dxyz[p]);
      }
      hypre_TFree(ssamg_data -> dxyz);

      hypre_FinalizeTiming(ssamg_data -> time_index);
      hypre_TFree(ssamg_data);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGSetTol( void       *ssamg_vdata,
                   HYPRE_Real  tol)
{
   hypre_SSAMGData *ssamg_data = (hypre_SSAMGData *) ssamg_vdata;

   hypre_SSAMGDataTol(ssamg_data) = tol;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGSetMaxIter( void       *ssamg_vdata,
                       HYPRE_Int   max_iter)
{
   hypre_SSAMGData *ssamg_data = (hypre_SSAMGData *) ssamg_vdata;

   hypre_SSAMGDataMaxIter(ssamg_data) = max_iter;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGSetMaxLevels( void       *ssamg_vdata,
                         HYPRE_Int   max_levels)
{
   hypre_SSAMGData *ssamg_data = (hypre_SSAMGData *) ssamg_vdata;

   hypre_SSAMGDataMaxLevels(ssamg_data) = max_levels;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGSetRelChange( void       *ssamg_vdata,
                         HYPRE_Real  rel_change)
{
   hypre_SSAMGData *ssamg_data = (hypre_SSAMGData *) ssamg_vdata;

   hypre_SSAMGDataRelChange(ssamg_data) = rel_change;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGSetZeroGuess( void       *ssamg_vdata,
                         HYPRE_Int   zero_guess)
{
   hypre_SSAMGData *ssamg_data = (hypre_SSAMGData *) ssamg_vdata;

   hypre_SSAMGDataZeroGuess(ssamg_data) = zero_guess;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGSetDxyz( void        *ssamg_vdata,
                    HYPRE_Int    nparts,
                    HYPRE_Real **dxyz       )
{
   hypre_SSAMGData *ssamg_data = (hypre_SSAMGData *) ssamg_vdata;

   HYPRE_Int        part;

   for (part = 0; part < nparts; part++)
   {
      (ssamg_data -> dxyz[part][0]) = dxyz[part][0];
      (ssamg_data -> dxyz[part][1]) = dxyz[part][1];
      (ssamg_data -> dxyz[part][2]) = dxyz[part][2];
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGSetRelaxType( void       *ssamg_vdata,
                         HYPRE_Int   relax_type)
{
   hypre_SSAMGData *ssamg_data = (hypre_SSAMGData *) ssamg_vdata;

   hypre_SSAMGDataRelaxType(ssamg_data) = relax_type;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGSetRelaxWeight( void        *ssamg_vdata,
                           HYPRE_Real   usr_relax_weight)
{
   hypre_SSAMGData *ssamg_data = (hypre_SSAMGData *) ssamg_vdata;

   hypre_SSAMGDataUsrRelaxWeight(ssamg_data) = usr_relax_weight;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGSetNumPreRelax( void       *ssamg_vdata,
                           HYPRE_Int   num_pre_relax)
{
   hypre_SSAMGData *ssamg_data = (hypre_SSAMGData *) ssamg_vdata;

   hypre_SSAMGDataNumPreRelax(ssamg_data) = num_pre_relax;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGSetNumPosRelax( void       *ssamg_vdata,
                           HYPRE_Int   num_pos_relax)
{
   hypre_SSAMGData *ssamg_data = (hypre_SSAMGData *) ssamg_vdata;

   hypre_SSAMGDataNumPosRelax(ssamg_data) = num_pos_relax;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGSetNumCoarseRelax( void       *ssamg_vdata,
                              HYPRE_Int   num_coarse_relax)
{
   hypre_SSAMGData *ssamg_data = (hypre_SSAMGData *) ssamg_vdata;

   hypre_SSAMGDataNumCoarseRelax(ssamg_data) = num_coarse_relax;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGSetPrintLevel( void       *ssamg_vdata,
                          HYPRE_Int   print_level)
{
   hypre_SSAMGData *ssamg_data = (hypre_SSAMGData *) ssamg_vdata;

   hypre_SSAMGDataPrintLevel(ssamg_data) = print_level;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGSetLogging( void       *ssamg_vdata,
                       HYPRE_Int   logging)
{
   hypre_SSAMGData *ssamg_data = (hypre_SSAMGData *) ssamg_vdata;

   hypre_SSAMGDataLogging(ssamg_data) = logging;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGPrintLogging( void *ssamg_vdata )
{
   hypre_SSAMGData   *ssamg_data     = (hypre_SSAMGData *) ssamg_vdata;
   MPI_Comm           comm           = (ssamg_data -> comm);
   HYPRE_Int          num_iterations = (ssamg_data -> num_iterations);
   HYPRE_Int          logging        = (ssamg_data -> logging);
   HYPRE_Int          print_level    = (ssamg_data -> print_level);
   HYPRE_Real        *norms          = (ssamg_data -> norms);
   HYPRE_Real        *rel_norms      = (ssamg_data -> rel_norms);
   HYPRE_Int          myid, i;
   HYPRE_Real         convr = 1.0;

   hypre_MPI_Comm_rank(comm, &myid);

   if (myid == 0)
   {
      if ((print_level > 0) && (logging > 1))
      {
         hypre_printf("Iters         ||r||_2   conv.rate  ||r||_2/||b||_2\n");
         hypre_printf("% 5d    %e    %f     %e\n", 0, norms[0], convr, rel_norms[0]);
         for (i = print_level; i < num_iterations; i = (i + print_level))
         {
            convr = norms[i] / norms[i-1];
            hypre_printf("% 5d    %e    %f     %e\n", i, norms[i], convr, rel_norms[i]);
         }

         if ((i != num_iterations) && (num_iterations > 0))
         {
            i = (num_iterations - 1);
            hypre_printf("% 5d    %e    %f     %e\n", i, norms[i], convr, rel_norms[i]);
         }
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGPrintStats( void *ssamg_vdata )
{
   hypre_SSAMGData        *ssamg_data    = (hypre_SSAMGData *) ssamg_vdata;
   MPI_Comm                comm          = hypre_SSAMGDataComm(ssamg_data);
   HYPRE_Int               num_levels    = hypre_SSAMGDataNumLevels(ssamg_data);
   HYPRE_Int               print_level   = hypre_SSAMGDataPrintLevel(ssamg_data);
   HYPRE_Int               relax_type    = hypre_SSAMGDataRelaxType(ssamg_data);
   HYPRE_Int               num_pre_relax = hypre_SSAMGDataNumPreRelax(ssamg_data);
   HYPRE_Int               num_pos_relax = hypre_SSAMGDataNumPosRelax(ssamg_data);
   HYPRE_Int               num_crelax    = hypre_SSAMGDataNumCoarseRelax(ssamg_data);
   HYPRE_Int               nparts        = hypre_SSAMGDataNParts(ssamg_data);
   HYPRE_Int             **cdir_l        = hypre_SSAMGDataCdir(ssamg_data);
   HYPRE_Real            **weights       = hypre_SSAMGDataRelaxWeights(ssamg_data);

   hypre_SStructMatrix   **A_l = hypre_SSAMGDataAl(ssamg_data);

   hypre_ParCSRMatrix     *umatrix;
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

   HYPRE_Int               min_entries;
   HYPRE_Int               max_entries;
   HYPRE_Complex           min_rowsum;
   HYPRE_Complex           max_rowsum;

   HYPRE_Complex           rowsum;
   HYPRE_Int               myid, i, ii, j, l, part;
   HYPRE_Int               entries;
   HYPRE_Int               chunk, chunk_size, chunk_last;
   HYPRE_Int               nparts_per_line = 8;
   HYPRE_Int               offset = 2;
   HYPRE_Int               ndigits[7] = {0, 0, 0, 9, 5, 4, 5};
   HYPRE_Int               header[4];
   HYPRE_Real              send_buffer[4];
   HYPRE_Real              recv_buffer[4];

   hypre_MPI_Comm_rank(comm, &myid);

   /* Update UMatrix info */
   for (l = 0; l < num_levels; l++)
   {
      umatrix = hypre_SStructMatrixParCSRMatrix(A_l[l]);
      hypre_ParCSRMatrixSetNumRownnz(umatrix);
      if (!hypre_ParCSRMatrixNumNonzeros(umatrix))
      {
         hypre_ParCSRMatrixSetNumNonzeros(umatrix);
      }
   }

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
   }

   for (l = 0; l < num_levels; l++)
   {
      umatrix = hypre_SStructMatrixParCSRMatrix(A_l[l]);
      diag    = hypre_ParCSRMatrixDiag(umatrix);
      offd    = hypre_ParCSRMatrixOffd(umatrix);
      diag_i  = hypre_CSRMatrixI(diag);
      offd_i  = hypre_CSRMatrixI(offd);
      diag_a  = hypre_CSRMatrixData(diag);
      offd_a  = hypre_CSRMatrixData(offd);

      diag_num_rownnz = hypre_CSRMatrixNumRownnz(diag);
      offd_num_rownnz = hypre_CSRMatrixNumRownnz(offd);
      diag_rownnz     = hypre_CSRMatrixRownnz(diag);
      offd_rownnz     = hypre_CSRMatrixRownnz(offd);
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

   if ((myid == 0) && (print_level > 0))
   {
      hypre_printf("\nSSAMG Setup Parameters:\n\n");

      /* Print coarsening direction */
      hypre_printf("Coarsening direction:\n\n");
      chunk_size = hypre_min(nparts, nparts_per_line);
      for (chunk = 0; chunk < nparts; chunk += chunk_size)
      {
         ndigits[0] = 4;
         hypre_printf("lev   ");
         chunk_last = hypre_min(chunk + chunk_size, nparts);
         for (part = chunk; part < chunk_last; part++)
         {
            hypre_printf("pt. %d  ", part);
            ndigits[0] += 7;
         }
         hypre_printf("\n");
         for (i = 0; i < ndigits[0]; i++) hypre_printf("%s", "=");
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

      /* Print Relaxation factor */
      if (relax_type > 0)
      {
         hypre_printf("Relaxation factors:\n\n");
         chunk_size = hypre_min(nparts, nparts_per_line);
         for (chunk = 0; chunk < nparts; chunk += chunk_size)
         {
            ndigits[0] = 4;
            hypre_printf("lev   ");
            chunk_last = hypre_min(chunk + chunk_size, nparts);
            for (part = chunk; part < chunk_last; part++)
            {
               hypre_printf("pt. %d  ", part);
               ndigits[0] += 7;
            }
            hypre_printf("\n");
            for (i = 0; i < ndigits[0]; i++) hypre_printf("%s", "=");
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

      /* Print UMatrix info */
      for (l = 0; l < num_levels; l++)
      {
         ndigits[1] = hypre_max(hypre_ndigits(global_num_rows[l]) + offset, ndigits[1]);
         ndigits[2] = hypre_max(hypre_ndigits(global_num_rownnz[l]) + offset, ndigits[2]);
         ndigits[3] = hypre_max(hypre_ndigits(global_num_nonzeros[l]) + offset, ndigits[3]);
         ndigits[4] = hypre_max(hypre_ndigits(global_min_entries[l]) + offset, ndigits[4]);
         ndigits[5] = hypre_max(hypre_ndigits(global_max_entries[l]) + offset, ndigits[5]);
         ndigits[6] = hypre_max(hypre_ndigits(global_avg_entries[l]) + offset, ndigits[6]);
      }

      header[0] = 3 + ndigits[1] + ndigits[2];
      header[1] = ndigits[3];
      header[2] = hypre_max(14, ndigits[4] + ndigits[5] + ndigits[6]);
      header[3] = header[0] + header[1] + header[2] + 22;

      /* Print first line of header */
      hypre_printf("UMatrix info:\n\n");
      hypre_printf("%*s", header[0], "nnz");
      hypre_printf("%*s", header[1], "nnz");
      hypre_printf("%*s", header[2], "entries/nnzrow");
      hypre_printf("%22s\n", "row sums");

      /* Print second line of header */
      hypre_printf("%s", "lev");
      hypre_printf("%*s", ndigits[1], "rows");
      hypre_printf("%*s", ndigits[2], "rows");
      hypre_printf("%*s", ndigits[3], "entries");
      hypre_printf("%*s", ndigits[4], "min");
      hypre_printf("%*s", ndigits[5], "max");
      hypre_printf("%*s", ndigits[6], "avg");
      hypre_printf("%11s %10s\n", "min", "max");

      /* Print third line of header */
      for (i = 0; i < header[3]; i++)
      {
         hypre_printf("%s", "=");
      }
      hypre_printf("\n");

      /* Print UMatrix info */
      for (l = 0; l < num_levels; l++)
      {
         hypre_printf("%3d", l);
         hypre_printf("%*d", ndigits[1], global_num_rows[l]);
         hypre_printf("%*d", ndigits[2], global_num_rownnz[l]);
         hypre_printf("%*d", ndigits[3], global_num_nonzeros[l]);
         hypre_printf("%*d", ndigits[4], global_min_entries[l]);
         hypre_printf("%*d", ndigits[5], global_max_entries[l]);
         hypre_printf("%*.1f", ndigits[6], global_avg_entries[l]);
         hypre_printf("%11.2e", global_min_rowsum[l]);
         hypre_printf("%11.2e", global_max_rowsum[l]);
         hypre_printf("\n");
      }
      hypre_printf("\n\n");

      /* SSAMG details */
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
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGGetNumIterations( void       *ssamg_vdata,
                             HYPRE_Int  *num_iterations)
{
   hypre_SSAMGData *ssamg_data = (hypre_SSAMGData *) ssamg_vdata;

   *num_iterations = hypre_SSAMGDataNumIterations(ssamg_data);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGGetFinalRelativeResidualNorm( void       *ssamg_vdata,
                                         HYPRE_Real *relative_residual_norm )
{
   hypre_SSAMGData *ssamg_data = (hypre_SSAMGData *) ssamg_vdata;

   HYPRE_Int        max_iter        = hypre_SSAMGDataMaxIter(ssamg_data);
   HYPRE_Int        num_iterations  = hypre_SSAMGDataNumIterations(ssamg_data);
   HYPRE_Int        logging         = hypre_SSAMGDataLogging(ssamg_data);
   HYPRE_Real      *rel_norms       = hypre_SSAMGDataRelNorms(ssamg_data);

   if (logging > 0)
   {
      if (max_iter == 0)
      {
         hypre_error_in_arg(1);
      }
      else if (num_iterations == max_iter)
      {
         *relative_residual_norm = rel_norms[num_iterations-1];
      }
      else
      {
         *relative_residual_norm = rel_norms[num_iterations];
      }
   }

   return hypre_error_flag;
}
