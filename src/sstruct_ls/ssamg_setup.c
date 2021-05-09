/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_sstruct_ls.h"
#include "ssamg.h"

#define DEBUG_SYMMETRY
#define DEBUG_MATMULT
//#define DEBUG_WITH_GLVIS

/*--------------------------------------------------------------------------
 *  TODO:
 *        1) Test full periodic problems. See what sys_pfmg does
 *        2) Implement HYPRE_SStructMatrixSetTranspose
 *        3) SetDxyz cannot be called by the user before SSAMGSetup because
 *           dxyz is being allocated here.
 *        4) Fix computation of cmaxsize. Should it vary across parts?
 *        5) Move "Initialize some data." to SSAMGCreate???
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGSetup( void                 *ssamg_vdata,
                  hypre_SStructMatrix  *A,
                  hypre_SStructVector  *b,
                  hypre_SStructVector  *x )
{
   hypre_SSAMGData       *ssamg_data = (hypre_SSAMGData *) ssamg_vdata;

   /* Data from the SStructMatrix */
   HYPRE_Int              ndim       = hypre_SStructMatrixNDim(A);
   hypre_SStructGraph    *graph      = hypre_SStructMatrixGraph(A);
   hypre_SStructGrid     *grid       = hypre_SStructGraphGrid(graph);

   /* Solver parameters */
   MPI_Comm               comm         = hypre_SSAMGDataComm(ssamg_data);
   HYPRE_Int              non_galerkin = hypre_SSAMGDataNonGalerkin(ssamg_data);
   HYPRE_Int              max_iter     = hypre_SSAMGDataMaxIter(ssamg_data);
   HYPRE_Int              max_levels   = hypre_SSAMGDataMaxLevels(ssamg_data);
   HYPRE_Int              relax_type   = hypre_SSAMGDataRelaxType(ssamg_data);
   HYPRE_Int              num_crelax   = hypre_SSAMGDataNumCoarseRelax(ssamg_data);
   HYPRE_Real           **dxyz         = hypre_SSAMGDataDxyz(ssamg_data);
   HYPRE_Int            **active_l;
   HYPRE_Int            **cdir_l;
   hypre_SStructGrid    **grid_l;

   /* Work data structures */
   hypre_SStructMatrix  **A_l  = (ssamg_data -> A_l);
   hypre_SStructMatrix  **P_l  = (ssamg_data -> P_l);
   hypre_SStructMatrix  **RT_l = (ssamg_data -> RT_l);
   hypre_SStructVector  **b_l  = (ssamg_data -> b_l);
   hypre_SStructVector  **x_l  = (ssamg_data -> x_l);
   hypre_SStructVector  **r_l  = (ssamg_data -> r_l);
   hypre_SStructVector  **e_l  = (ssamg_data -> e_l);
   hypre_SStructVector  **tx_l = (ssamg_data -> tx_l);

   /* Data pointers */
   void                 **relax_data_l    = (ssamg_data -> relax_data_l);
   void                 **matvec_data_l   = (ssamg_data -> matvec_data_l);
   void                 **restrict_data_l = (ssamg_data -> restrict_data_l);
   void                 **interp_data_l   = (ssamg_data -> interp_data_l);

   HYPRE_Int             *dxyz_flag;
   HYPRE_Real           **relax_weights;
   HYPRE_Int              d, l;
   HYPRE_Int              nparts;
   HYPRE_Int              num_levels;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   /*-----------------------------------------------------
    * Sanity checks
    *-----------------------------------------------------*/
   if (hypre_SStructMatrixObjectType(A) != HYPRE_SSTRUCT)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Matrix is not HYPRE_SSTRUCT");
      HYPRE_ANNOTATE_FUNC_END;

      return hypre_error_flag;
   }

   if (hypre_SStructVectorObjectType(x) != HYPRE_SSTRUCT)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "LHS is not HYPRE_SSTRUCT");
      HYPRE_ANNOTATE_FUNC_END;

      return hypre_error_flag;
   }

   if (hypre_SStructVectorObjectType(b) != HYPRE_SSTRUCT)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "RHS is not HYPRE_SSTRUCT");
      HYPRE_ANNOTATE_FUNC_END;

      return hypre_error_flag;
   }

   /*-----------------------------------------------------
    * Initialize some data.
    *-----------------------------------------------------*/
   nparts    = hypre_SStructMatrixNParts(A);
   dxyz_flag = hypre_TAlloc(HYPRE_Int, nparts, HYPRE_MEMORY_HOST);
   for (d = 0; d < ndim; d++)
   {
      dxyz[d] = hypre_CTAlloc(HYPRE_Real, nparts, HYPRE_MEMORY_HOST);
      hypre_SSAMGDataDxyzD(ssamg_data, d) = dxyz[d];
   }
   hypre_SSAMGDataNParts(ssamg_data) = nparts;

   /* Compute Maximum number of multigrid levels */
   hypre_SSAMGComputeMaxLevels(grid, &max_levels);
   hypre_SSAMGDataMaxLevels(ssamg_data) = max_levels;

   /* Compute dxyz for each part */
   hypre_SSAMGComputeDxyz(A, dxyz, dxyz_flag);

   /* Compute coarsening direction and active levels for relaxation */
   hypre_SSAMGCoarsen(ssamg_vdata, grid, dxyz_flag, dxyz);

   /* Compute maximum number of iterations in coarsest grid if requested */
   hypre_SSAMGComputeNumCoarseRelax(ssamg_vdata);

   /* Get info from ssamg_data */
   cdir_l        = hypre_SSAMGDataCdir(ssamg_data);
   grid_l        = hypre_SSAMGDataGridl(ssamg_data);
   active_l      = hypre_SSAMGDataActivel(ssamg_data);
   num_crelax    = hypre_SSAMGDataNumCoarseRelax(ssamg_data);
   relax_weights = hypre_SSAMGDataRelaxWeights(ssamg_data);

   /*-----------------------------------------------------
    * Set up matrix and vector structures
    *-----------------------------------------------------*/

   num_levels = hypre_SSAMGDataNumLevels(ssamg_data);
   A_l  = hypre_TAlloc(hypre_SStructMatrix *, num_levels, HYPRE_MEMORY_HOST);
   P_l  = hypre_TAlloc(hypre_SStructMatrix *, num_levels - 1, HYPRE_MEMORY_HOST);
   RT_l = hypre_TAlloc(hypre_SStructMatrix *, num_levels - 1, HYPRE_MEMORY_HOST);
   b_l  = hypre_TAlloc(hypre_SStructVector *, num_levels, HYPRE_MEMORY_HOST);
   x_l  = hypre_TAlloc(hypre_SStructVector *, num_levels, HYPRE_MEMORY_HOST);
   tx_l = hypre_TAlloc(hypre_SStructVector *, num_levels, HYPRE_MEMORY_HOST);
   r_l  = tx_l;
   e_l  = tx_l;

   hypre_SStructMatrixRef(A, &A_l[0]);
   hypre_SStructVectorRef(b, &b_l[0]);
   hypre_SStructVectorRef(x, &x_l[0]);

   HYPRE_SStructVectorCreate(comm, grid_l[0], &tx_l[0]);
   HYPRE_SStructVectorInitialize(tx_l[0]);
   HYPRE_SStructVectorAssemble(tx_l[0]);

   /* Compute interpolation, restriction and coarse grids */
   for (l = 0; l < (num_levels - 1); l++)
   {
      HYPRE_ANNOTATE_MGLEVEL_BEGIN(l);

      // Build prolongation matrix
      P_l[l]  = hypre_SSAMGCreateInterpOp(A_l[l], grid_l[l+1], cdir_l[l]);
      //HYPRE_SStructMatrixSetTranspose(P_l[l], 1);
      hypre_SSAMGSetupInterpOp(A_l[l], cdir_l[l], P_l[l]);

      // Build restriction matrix
      hypre_SStructMatrixRef(P_l[l], &RT_l[l]);

      // Compute coarse matrix
      hypre_SSAMGComputeRAP(A_l[l], P_l[l], grid_l[l+1], cdir_l[l], non_galerkin, &A_l[l+1]);

      // Build SStructVectors
      HYPRE_SStructVectorCreate(comm, grid_l[l+1], &b_l[l+1]);
      HYPRE_SStructVectorInitialize(b_l[l+1]);
      HYPRE_SStructVectorAssemble(b_l[l+1]);

      HYPRE_SStructVectorCreate(comm, grid_l[l+1], &x_l[l+1]);
      HYPRE_SStructVectorInitialize(x_l[l+1]);
      HYPRE_SStructVectorAssemble(x_l[l+1]);

      HYPRE_SStructVectorCreate(comm, grid_l[l+1], &tx_l[l+1]);
      HYPRE_SStructVectorInitialize(tx_l[l+1]);
      HYPRE_SStructVectorAssemble(tx_l[l+1]);

      HYPRE_ANNOTATE_MGLEVEL_END(l);
   }

   (ssamg_data -> A_l)  = A_l;
   (ssamg_data -> P_l)  = P_l;
   (ssamg_data -> RT_l) = RT_l;
   (ssamg_data -> b_l)  = b_l;
   (ssamg_data -> x_l)  = x_l;
   (ssamg_data -> tx_l) = tx_l;
   (ssamg_data -> r_l)  = r_l;
   (ssamg_data -> e_l)  = e_l;

   /*-----------------------------------------------------
    * Set up multigrid operators and call setup routines
    *-----------------------------------------------------*/

   relax_data_l    = hypre_TAlloc(void *, num_levels, HYPRE_MEMORY_HOST);
   matvec_data_l   = hypre_TAlloc(void *, num_levels, HYPRE_MEMORY_HOST);
   restrict_data_l = hypre_TAlloc(void *, num_levels, HYPRE_MEMORY_HOST);
   interp_data_l   = hypre_TAlloc(void *, num_levels, HYPRE_MEMORY_HOST);

   for (l = 0; l < (num_levels - 1); l++)
   {
      hypre_SStructMatvecCreate(&matvec_data_l[l]);
      hypre_SStructMatvecSetup(matvec_data_l[l], A_l[l], x_l[l]);

      hypre_SStructMatvecCreate(&interp_data_l[l]);
      hypre_SStructMatvecSetup(interp_data_l[l], P_l[l], x_l[l+1]);

      hypre_SStructMatvecCreate(&restrict_data_l[l]);
      hypre_SStructMatvecSetTranspose(restrict_data_l[l], 1);
      hypre_SStructMatvecSetup(restrict_data_l[l], RT_l[l], x_l[l]);

      hypre_SSAMGRelaxCreate(comm, nparts, &relax_data_l[l]);
      hypre_SSAMGRelaxSetTol(relax_data_l[l], 0.0);
      hypre_SSAMGRelaxSetWeights(relax_data_l[l], relax_weights[l]);
      hypre_SSAMGRelaxSetActiveParts(relax_data_l[l], active_l[l]);
      hypre_SSAMGRelaxSetType(relax_data_l[l], relax_type);
      hypre_SSAMGRelaxSetTempVec(relax_data_l[l], tx_l[l]);
      hypre_SSAMGRelaxSetMatvecData(relax_data_l[l], matvec_data_l[l]);
      hypre_SSAMGRelaxSetup(relax_data_l[l], A_l[l], b_l[l], x_l[l]);
   }

   /* set up remaining operations for the coarse grid */
   l = (num_levels - 1);
   hypre_SStructMatvecCreate(&matvec_data_l[l]);
   hypre_SStructMatvecSetup(matvec_data_l[l], A_l[l], x_l[l]);
   hypre_SSAMGRelaxCreate(comm, nparts, &relax_data_l[l]);
   hypre_SSAMGRelaxSetTol(relax_data_l[l], 0.0);
   hypre_SSAMGRelaxSetWeights(relax_data_l[l], relax_weights[l]);
   hypre_SSAMGRelaxSetActiveParts(relax_data_l[l], active_l[l]);
   hypre_SSAMGRelaxSetType(relax_data_l[l], 0);
   hypre_SSAMGRelaxSetMaxIter(relax_data_l[l], num_crelax);
   hypre_SSAMGRelaxSetTempVec(relax_data_l[l], tx_l[l]);
   hypre_SSAMGRelaxSetMatvecData(relax_data_l[l], matvec_data_l[l]);
   hypre_SSAMGRelaxSetup(relax_data_l[l], A_l[l], b_l[l], x_l[l]);

   (ssamg_data -> relax_data_l)    = relax_data_l;
   (ssamg_data -> matvec_data_l)   = matvec_data_l;
   (ssamg_data -> restrict_data_l) = restrict_data_l;
   (ssamg_data -> interp_data_l)   = interp_data_l;

   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/

   if ((ssamg_data -> logging) > 0)
   {
      max_iter = (ssamg_data -> max_iter);
      (ssamg_data -> norms)     = hypre_CTAlloc(HYPRE_Real, max_iter + 1, HYPRE_MEMORY_HOST);
      (ssamg_data -> rel_norms) = hypre_CTAlloc(HYPRE_Real, max_iter + 1, HYPRE_MEMORY_HOST);
   }

   /* Print statistics */
   hypre_SSAMGPrintStats(ssamg_vdata);

   /* Free memory */
   hypre_TFree(dxyz_flag, HYPRE_MEMORY_HOST);

#ifdef DEBUG_SETUP
   hypre_SStructVector  **ones_l;
   hypre_SStructVector  **Aones_l;
   hypre_SStructVector  **Pones_l;
   hypre_SStructPGrid    *pgrid;
   HYPRE_Int              mypid, part;
   char                   filename[255];

   hypre_MPI_Comm_rank(hypre_SStructMatrixComm(A), &mypid);

   ones_l  = hypre_TAlloc(hypre_SStructVector *, num_levels, HYPRE_MEMORY_HOST);
   Aones_l = hypre_TAlloc(hypre_SStructVector *, num_levels, HYPRE_MEMORY_HOST);
   Pones_l = hypre_TAlloc(hypre_SStructVector *, num_levels - 1, HYPRE_MEMORY_HOST);

   /* Print fine level grid */
   hypre_sprintf(filename, "ssgrid.l%02d", 0);
#ifdef DEBUG_WITH_GLVIS
   hypre_SStructGridPrintGLVis(grid_l[0], filename, NULL, NULL);
#else
   hypre_SStructGridPrint(grid_l[0], filename);
#endif

   /* Print part boundary data */
   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGridPGrid(grid_l[0], part);

      hypre_sprintf(filename, "pbnd_boxa.l%02d.p%02d", 0, part);
      hypre_BoxArrayArrayPrint(hypre_SStructGridComm(grid_l[0]),
                               filename,
                               hypre_SStructPGridPBndBoxArrayArray(pgrid, 0));
   }

   /* Print fine level matrix */
   hypre_sprintf(filename, "ssamg_A.l%02d", 0);
   HYPRE_SStructMatrixPrint(filename, A_l[0], 0);

   /* compute Aones = A.1 */
   HYPRE_SStructVectorCreate(comm, grid_l[0], &ones_l[0]);
   HYPRE_SStructVectorInitialize(ones_l[0]);
   HYPRE_SStructVectorSetConstantValues(ones_l[0], 1.0);
   HYPRE_SStructVectorAssemble(ones_l[0]);
   HYPRE_SStructVectorCreate(comm, grid_l[0], &Aones_l[0]);
   HYPRE_SStructVectorInitialize(Aones_l[0]);
   HYPRE_SStructVectorAssemble(Aones_l[0]);
   hypre_SStructMatvecCompute(matvec_data_l[0], 1.0, A_l[0], ones_l[0],
                              0.0, Aones_l[0], Aones_l[0]);

   /* Print Aones */
   hypre_sprintf(filename, "ssamg_Aones.l%02d", 0);
#ifdef DEBUG_WITH_GLVIS
   HYPRE_SStructVectorPrintGLVis(Aones_l[0], filename);
#else
   HYPRE_SStructVectorPrint(filename, Aones_l[0], 0);
#endif

   for (l = 0; l < (num_levels - 1); l++)
   {
      /* Print coarse grids */
      hypre_sprintf(filename, "ssgrid.l%02d", l+1);
#ifdef DEBUG_WITH_GLVIS
      hypre_SStructGridPrintGLVis(grid_l[l+1], filename, NULL, NULL);
#else
      hypre_SStructGridPrint(grid_l[l+1], filename);
#endif

      /* Print part boundary data */
      for (part = 0; part < nparts; part++)
      {
         pgrid = hypre_SStructGridPGrid(grid_l[l+1], part);

         hypre_sprintf(filename, "pbnd_boxa.l%02d.p%02d", l+1, part);
         hypre_BoxArrayArrayPrint(hypre_SStructGridComm(grid_l[l+1]),
                                  filename,
                                  hypre_SStructPGridPBndBoxArrayArray(pgrid, 0));
      }

      /* Print coarse matrices */
      hypre_sprintf(filename, "ssamg_A.l%02d", l+1);
      HYPRE_SStructMatrixPrint(filename, A_l[l+1], 0);

      /* Print interpolation matrix */
      hypre_sprintf(filename, "ssamg_P.l%02d", l);
      HYPRE_SStructMatrixPrint(filename, P_l[l], 0);

      /* compute Pones = P.1 */
      HYPRE_SStructVectorCreate(comm, grid_l[l+1], &ones_l[l+1]);
      HYPRE_SStructVectorInitialize(ones_l[l+1]);
      HYPRE_SStructVectorSetConstantValues(ones_l[l+1], 1.0);
      HYPRE_SStructVectorAssemble(ones_l[l+1]);
      HYPRE_SStructVectorCreate(comm, grid_l[l], &Pones_l[l]);
      HYPRE_SStructVectorInitialize(Pones_l[l]);
      HYPRE_SStructVectorAssemble(Pones_l[l]);
      hypre_SStructMatvecCompute(interp_data_l[l], 1.0, P_l[l], ones_l[l],
                                 0.0, Pones_l[l], Pones_l[l]);

      /* Print Pones */
      hypre_sprintf(filename, "ssamg_Pones.l%02d", l);
#ifdef DEBUG_WITH_GLVIS
      HYPRE_SStructVectorPrintGLVis(Pones_l[l], filename);
#else
      HYPRE_SStructVectorPrint(filename, Pones_l[l], 0);
#endif

      /* compute Aones = A.1 */
      HYPRE_SStructVectorCreate(comm, grid_l[l+1], &Aones_l[l+1]);
      HYPRE_SStructVectorInitialize(Aones_l[l+1]);
      HYPRE_SStructVectorAssemble(Aones_l[l+1]);
      hypre_SStructMatvecCompute(matvec_data_l[l+1], 1.0, A_l[l+1], ones_l[l+1],
                                 0.0, Aones_l[l+1], Aones_l[l+1]);

      /* Print Aones */
      hypre_sprintf(filename, "ssamg_Aones.l%02d", l+1);
#ifdef DEBUG_WITH_GLVIS
      HYPRE_SStructVectorPrintGLVis(Aones_l[l+1], filename);
#else
      HYPRE_SStructVectorPrint(filename, Aones_l[l+1], 0);
#endif
   }

   HYPRE_SStructVectorDestroy(ones_l[0]);
   HYPRE_SStructVectorDestroy(Aones_l[0]);
   for (l = 0; l < (num_levels - 1); l++)
   {
      HYPRE_SStructVectorDestroy(ones_l[l+1]);
      HYPRE_SStructVectorDestroy(Aones_l[l+1]);
      HYPRE_SStructVectorDestroy(Pones_l[l]);
   }
   hypre_TFree(ones_l, HYPRE_MEMORY_HOST);
   hypre_TFree(Pones_l, HYPRE_MEMORY_HOST);
   hypre_TFree(Aones_l, HYPRE_MEMORY_HOST);

   /* Compute Frobenius norm of (A - A^T) */
#ifdef DEBUG_SYMMETRY
   {
      HYPRE_IJMatrix         ij_A, ij_AT, ij_B;
      HYPRE_Real             B_norm;

      for (l = 0; l < num_levels; l++)
      {
         HYPRE_SStructMatrixToIJMatrix(A_l[l], &ij_A);
         HYPRE_IJMatrixTranspose(ij_A, &ij_AT);
         HYPRE_IJMatrixAdd(1.0, ij_A, -1.0, ij_AT, &ij_B);
         HYPRE_IJMatrixNorm(ij_B, &B_norm);
         if (!mypid)
         {
            hypre_printf("Frobenius norm (A[%02d] - A[%02d]^T) = %20.15e\n", l, l, B_norm);
         }

         /* Print matrices */
         hypre_sprintf(filename, "ssamg_ijA.l%02d", l);
         HYPRE_IJMatrixPrint(ij_A, filename);
         hypre_sprintf(filename, "ssamg_ijB.l%02d", l);
         HYPRE_IJMatrixPrint(ij_B, filename);

         /* Free memory */
         HYPRE_IJMatrixDestroy(ij_A);
         HYPRE_IJMatrixDestroy(ij_AT);
         HYPRE_IJMatrixDestroy(ij_B);
      }
   }
#endif /* DEBUG_SYMMETRY */

#ifdef DEBUG_MATMULT
   {
      HYPRE_IJMatrix       ij_A[2], ij_P;
      hypre_ParCSRMatrix  *par_A[2], *par_P, *par_AP, *par_RAP, *par_B;
      HYPRE_Real           norm;

      HYPRE_SStructMatrixToIJMatrix(A_l[0], &ij_A[0]);
      HYPRE_IJMatrixGetObject(ij_A[0], (void **) &par_A[0]);
      for (l = 0; l < num_levels-1; l++)
      {
         if (!mypid) hypre_printf("Converting A[%02d]\n", l);
         HYPRE_SStructMatrixToIJMatrix(A_l[l+1], &ij_A[1]);
         if (!mypid) hypre_printf("Converting P[%02d]\n", l);
         HYPRE_SStructMatrixToIJMatrix(P_l[l], &ij_P);

         HYPRE_IJMatrixGetObject(ij_A[1], (void **) &par_A[1]);
         HYPRE_IJMatrixGetObject(ij_P, (void **) &par_P);

         par_AP  = hypre_ParMatmul(par_A[0], par_P);
         par_RAP = hypre_ParTMatmul(par_P, par_AP);

         hypre_ParcsrAdd(1.0, par_RAP, -1.0, par_A[1], &par_B);
         norm = hypre_ParCSRMatrixFnorm(par_B);
         if (!mypid)
         {
            hypre_printf("Frobenius norm (RAP_par[%02d] - RAP_ss[%02d]^T) = %20.15e\n", l, l, norm);
         }

         /* Print matrices */
         hypre_sprintf(filename, "ssamg_ijP.l%02d", l);
         if (!mypid) hypre_printf("Printing %s\n", filename);
         HYPRE_IJMatrixPrint(ij_P, filename);

         hypre_sprintf(filename, "ssamg_ijRAPdiff.l%02d", l+1);
         if (!mypid) hypre_printf("Printing %s\n", filename);
         hypre_ParCSRMatrixPrintIJ(par_B, 0, 0, filename);

         HYPRE_IJMatrixDestroy(ij_A[0]);
         ij_A[0]  = ij_A[1];
         par_A[0] = par_A[1];

         HYPRE_IJMatrixDestroy(ij_P);
         hypre_ParCSRMatrixDestroy(par_AP);
         hypre_ParCSRMatrixDestroy(par_RAP);
         hypre_ParCSRMatrixDestroy(par_B);
      }
      HYPRE_IJMatrixDestroy(ij_A[0]);
   }
#endif // ifdef (DEBUG_MATMULT)
#endif // ifdef (DEBUG_SETUP)

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SSAMGComputeCoarseMaxIter
 *
 * Computes maximum number of iterations for the coarsest grid
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGComputeNumCoarseRelax( void *ssamg_vdata )
{
   hypre_SSAMGData      *ssamg_data  = (hypre_SSAMGData *) ssamg_vdata;
   HYPRE_Int             nparts      = hypre_SSAMGDataNParts(ssamg_data);
   HYPRE_Int             num_levels  = hypre_SSAMGDataNumLevels(ssamg_data);
   HYPRE_Int             num_crelax  = hypre_SSAMGDataNumCoarseRelax(ssamg_data);
   hypre_SStructGrid    *cgrid       = hypre_SSAMGDataGridl(ssamg_data)[num_levels - 1];

   hypre_Box            *bbox;
   hypre_SStructPGrid   *pgrid;
   hypre_StructGrid     *sgrid;

   HYPRE_Int             part, cmax_size, max_work;

   if (num_crelax < 0)
   {
      /* do no more work on the coarsest grid than the cost of a V-cycle
       * (estimating roughly 4 communications per V-cycle level) */
      max_work = 4*num_levels;

      /* do sweeps proportional to the coarsest grid size */
      cmax_size = 0;
      for (part = 0; part < nparts; part++)
      {
         pgrid = hypre_SStructGridPGrid(cgrid, part);
         sgrid = hypre_SStructPGridCellSGrid(pgrid);
         bbox  = hypre_StructGridBoundingBox(sgrid);

         cmax_size = hypre_max(cmax_size, hypre_BoxMaxSize(bbox));
      }
      hypre_SSAMGDataNumCoarseRelax(ssamg_data) = hypre_min(max_work, cmax_size);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SSAMGComputeMaxLevels
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGComputeMaxLevels( hypre_SStructGrid  *grid,
                             HYPRE_Int          *max_levels )
{
   hypre_SStructPGrid   *pgrid;
   hypre_StructGrid     *sgrid;

   HYPRE_Int             max_levels_in;
   HYPRE_Int             max_levels_out;
   HYPRE_Int             max_levels_part;
   HYPRE_Int             part, nparts;

   nparts = hypre_SStructGridNParts(grid);

   max_levels_in  = *max_levels;
   max_levels_out = 0;
   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGridPGrid(grid, part);
      sgrid = hypre_SStructPGridCellSGrid(pgrid);

      hypre_PFMGComputeMaxLevels(sgrid, &max_levels_part);

      max_levels_out = hypre_max(max_levels_part, max_levels_out);
   }

   *max_levels = max_levels_out;
   if (max_levels_in > 0)
   {
      *max_levels = hypre_min(max_levels_out, max_levels_in);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SSAMGComputeDxyz computes dxyz for each part independently.
 *
 * TODO: Before implementing, check if this makes sense
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGComputeDxyz( hypre_SStructMatrix  *A,
                        HYPRE_Real          **dxyz,
                        HYPRE_Int            *dxyz_flag )
{
   hypre_SStructPMatrix  *pmatrix;
   hypre_StructMatrix    *smatrix;

   HYPRE_Int              d, var, part, nparts, nvars;
   HYPRE_Int              ndim;
   HYPRE_Real             sys_dxyz[3] = {0.0, 0.0, 0.0};

   /*--------------------------------------------------------
    * Allocate arrays for mesh sizes for each diagonal block
    *--------------------------------------------------------*/
   nparts = hypre_SStructMatrixNParts(A);
   ndim   = hypre_SStructMatrixNDim(A);

   for (part = 0; part < nparts; part++)
   {
      pmatrix = hypre_SStructMatrixPMatrix(A, part);
      nvars   = hypre_SStructPMatrixNVars(pmatrix);

      for (var = 0; var < nvars; var++)
      {
         smatrix = hypre_SStructPMatrixSMatrix(pmatrix, var, var);
         hypre_PFMGComputeDxyz(smatrix, sys_dxyz, &dxyz_flag[part]);

         for (d = 0; d < ndim; d++)
         {
            dxyz[d][part] += sys_dxyz[d];
            sys_dxyz[d] = 0.0;
         }
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGCoarsen( void               *ssamg_vdata,
                    hypre_SStructGrid  *grid,
                    HYPRE_Int          *dxyz_flag,
                    HYPRE_Real        **dxyz )
{
   hypre_SSAMGData      *ssamg_data  = (hypre_SSAMGData *) ssamg_vdata;
   HYPRE_Int             ndim        = hypre_SStructGridNDim(grid);
   HYPRE_Int             skip_relax  = hypre_SSAMGDataSkipRelax(ssamg_data);
   HYPRE_Int             max_levels  = hypre_SSAMGDataMaxLevels(ssamg_data);
   HYPRE_Int             nparts      = hypre_SSAMGDataNParts(ssamg_data);
   HYPRE_Real            usr_relax   = hypre_SSAMGDataUsrRelaxWeight(ssamg_data);

   HYPRE_Int           **active_l;
   hypre_SStructGrid   **grid_l;
   hypre_SStructPGrid   *pgrid;
   hypre_StructGrid     *sgrid;
   hypre_Box           **cbox;

   HYPRE_Int           **cdir_l;
   HYPRE_Real          **weights;
   hypre_Index          *periodic;
   hypre_Index          *strides;
   hypre_Index          *coarsen;

   hypre_Index           cindex;
   hypre_Index           zero;
   HYPRE_Int             num_levels;
   HYPRE_Int             part;
   HYPRE_Int             l, d, cdir;
   HYPRE_Int             coarse_flag;
   HYPRE_Int             cbox_imin, cbox_imax;
   HYPRE_Real            min_dxyz;
   HYPRE_Real            alpha, beta;

   /* Allocate data */
   grid_l   = hypre_TAlloc(hypre_SStructGrid *, max_levels, HYPRE_MEMORY_HOST);
   active_l = hypre_TAlloc(HYPRE_Int *, max_levels, HYPRE_MEMORY_HOST);
   cdir_l   = hypre_TAlloc(HYPRE_Int *, max_levels, HYPRE_MEMORY_HOST);
   weights  = hypre_TAlloc(HYPRE_Real *, max_levels, HYPRE_MEMORY_HOST);
   cbox     = hypre_TAlloc(hypre_Box *, nparts, HYPRE_MEMORY_HOST);
   periodic = hypre_TAlloc(hypre_Index, nparts, HYPRE_MEMORY_HOST);
   strides  = hypre_TAlloc(hypre_Index, nparts, HYPRE_MEMORY_HOST);
   coarsen  = hypre_TAlloc(hypre_Index, nparts, HYPRE_MEMORY_HOST);

   hypre_SStructGridRef(grid, &grid_l[0]);

   /* Initalize data */
   for (l = 0; l < max_levels; l++)
   {
      weights[l]  = hypre_CTAlloc(HYPRE_Real, nparts, HYPRE_MEMORY_HOST);
      if (usr_relax > 0.0)
      {
         for (part = 0; part < nparts; part++)
         {
            weights[l][part] = usr_relax;
         }
      }

      active_l[l] = hypre_CTAlloc(HYPRE_Int, nparts, HYPRE_MEMORY_HOST);
      for (part = 0; part < nparts; part++)
      {
         active_l[l][part] = 1;
      }
   }

   for (part = 0; part < nparts; part++)
   {
      hypre_SetIndex(strides[part], 1);

      /* Force relaxation on finest grid */
      hypre_SetIndex(coarsen[part], 1);
   }
   hypre_SetIndex(zero, 0);

   /* Get grid bounding box and periodicity data */
   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGridPGrid(grid_l[0], part);
      sgrid = hypre_SStructPGridCellSGrid(pgrid);
      cbox[part] = hypre_BoxClone(hypre_StructGridBoundingBox(sgrid));

      hypre_CopyIndex(hypre_StructGridPeriodic(sgrid), periodic[part]);
   }

   for (l = 0; l < max_levels; l++)
   {
      cdir_l[l] = hypre_TAlloc(HYPRE_Int, nparts, HYPRE_MEMORY_HOST);
      for (part = 0; part < nparts; part++)
      {
         cdir_l[l][part] = -1;
      }

      coarse_flag = 0;
      for (part = 0; part < nparts; part++)
      {
         /* Initialize min_dxyz */
         min_dxyz = 1;
         for (d = 0; d < ndim; d++)
         {
            min_dxyz += dxyz[d][part];
         }

         /* Determine cdir */
         cdir = -1; alpha = 0.0;
         for (d = 0; d < ndim; d++)
         {
            cbox_imax = hypre_BoxIMaxD(cbox[part], d);
            cbox_imin = hypre_BoxIMinD(cbox[part], d);

            if ((cbox_imax > cbox_imin) && (dxyz[d][part] < min_dxyz))
            {
               min_dxyz = dxyz[d][part];
               cdir = d;
            }
            alpha += 1.0/(dxyz[d][part]*dxyz[d][part]);
         }

         /* Change relax_weights */
         if (usr_relax <= 0.0)
         {
            beta = 0.0;
            if (dxyz_flag[part] || (ndim == 1))
            {
               weights[l][part] = 2.0/3.0;
            }
            else
            {
               for (d = 0; d < ndim; d++)
               {
                  if (d != cdir)
                  {
                     beta += 1.0/(dxyz[d][part]*dxyz[d][part]);
                  }
               }

               /* determine level Jacobi weights */
               weights[l][part] = 2.0/(3.0 - beta/alpha);
            }
         }

         if ((cdir > -1) && (l < (max_levels - 1)))
         {
            coarse_flag = 1;
            cdir_l[l][part] = cdir;

            /* don't coarsen if a periodic direction and not divisible by 2 */
            if ((periodic[part][cdir]) && (periodic[part][cdir] % 2))
            {
               cdir_l[l][part] = -1;
               continue;
            }

            if (hypre_IndexD(coarsen[part], cdir) != 0)
            {
               /* coarsened previously in this direction, relax level l */
               active_l[l][part] = 1;
               //hypre_IndexD(coarsen[part], cdir) = 0;
               hypre_SetIndex(coarsen[part], 0);
            }
            else
            {
               if (skip_relax)
               {
                  active_l[l][part] = 0;
               }
            }
            hypre_IndexD(coarsen[part], cdir) = 1;

            /* set cindex and stride */
            hypre_SetIndex(cindex, 0);
            hypre_SetIndex(strides[part], 1);
            hypre_IndexD(strides[part], cdir) = 2;

            /* update dxyz and coarsen cbox*/
            dxyz[cdir][part] *= 2;
            hypre_ProjectBox(cbox[part], cindex, strides[part]);
            hypre_StructMapFineToCoarse(hypre_BoxIMin(cbox[part]), cindex, strides[part],
                                        hypre_BoxIMin(cbox[part]));
            hypre_StructMapFineToCoarse(hypre_BoxIMax(cbox[part]), cindex, strides[part],
                                        hypre_BoxIMax(cbox[part]));

            /* update periodic */
            periodic[part][cdir] /= 2;
         }
         else
         {
            /* Update coarse grid relax weight */
            if (max_levels > 1)
            {
               weights[l][part] = 1.0;
            }
         }
      } /* loop on parts */

      // If there's no part to be coarsened, exit loop
      if (!coarse_flag)
      {
         num_levels = l + 1;
         break;
      }

      // Compute the coarsened SStructGrid object
      hypre_SStructGridCoarsen(grid_l[l], zero, strides, periodic, &grid_l[l+1]);
   } /* loop on levels */

   /* Free memory */
   for (part = 0; part < nparts; part++)
   {
      hypre_BoxDestroy(cbox[part]);
   }
   hypre_TFree(cbox, HYPRE_MEMORY_HOST);
   hypre_TFree(periodic, HYPRE_MEMORY_HOST);
   hypre_TFree(strides, HYPRE_MEMORY_HOST);
   hypre_TFree(coarsen, HYPRE_MEMORY_HOST);

   /* Output */
   hypre_SSAMGDataCdir(ssamg_data)         = cdir_l;
   hypre_SSAMGDataGridl(ssamg_data)        = grid_l;
   hypre_SSAMGDataActivel(ssamg_data)      = active_l;
   hypre_SSAMGDataNumLevels(ssamg_data)    = num_levels;
   hypre_SSAMGDataRelaxWeights(ssamg_data) = weights;

   return hypre_error_flag;
}
