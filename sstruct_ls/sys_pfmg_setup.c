/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 *
 *****************************************************************************/

#include "headers.h"
#include "sys_pfmg.h"

#define DEBUG 0

#define hypre_PFMGSetCIndex(cdir, cindex) \
{\
   hypre_SetIndex(cindex, 0, 0, 0);\
   hypre_IndexD(cindex, cdir) = 0;\
}

#define hypre_PFMGSetFIndex(cdir, findex) \
{\
   hypre_SetIndex(findex, 0, 0, 0);\
   hypre_IndexD(findex, cdir) = 1;\
}

#define hypre_PFMGSetStride(cdir, stride) \
{\
   hypre_SetIndex(stride, 1, 1, 1);\
   hypre_IndexD(stride, cdir) = 2;\
}

/*--------------------------------------------------------------------------
 * hypre_SysPFMGSetup
 *--------------------------------------------------------------------------*/

int
hypre_SysPFMGSetup( void                 *sys_pfmg_vdata,
                    hypre_SStructMatrix  *A_in,
                    hypre_SStructVector  *b_in,
                    hypre_SStructVector  *x_in        )
{
   hypre_SysPFMGData    *sys_pfmg_data = sys_pfmg_vdata;

   MPI_Comm              comm = (sys_pfmg_data -> comm);
                     
   hypre_SStructPMatrix *A;
   hypre_SStructPVector *b;
   hypre_SStructPVector *x;

   int                   relax_type = (sys_pfmg_data -> relax_type);
   int                   skip_relax = (sys_pfmg_data -> skip_relax);
   double               *dxyz       = (sys_pfmg_data -> dxyz);
                     
   int                   max_iter;
   int                   max_levels;
                      
   int                   num_levels;
                     
   hypre_Index           cindex;
   hypre_Index           findex;
   hypre_Index           stride;

   hypre_Index           coarsen;

   int                    *cdir_l;
   int                    *active_l;
   hypre_SStructPGrid    **grid_l;
   hypre_SStructPGrid    **P_grid_l;
                    
   hypre_SStructPMatrix  **A_l;
   hypre_SStructPMatrix  **P_l;
   hypre_SStructPMatrix  **RT_l;
   hypre_SStructPVector  **b_l;
   hypre_SStructPVector  **x_l;

   /* temp vectors */
   hypre_SStructPVector  **tx_l;
   hypre_SStructPVector  **r_l;
   hypre_SStructPVector  **e_l;

   void                **relax_data_l;
   void                **matvec_data_l;
   void                **restrict_data_l;
   void                **interp_data_l;

   hypre_SStructPGrid     *grid;
   hypre_StructGrid       *sgrid;
   int                     dim;
   int                     full_periodic;

   hypre_Box            *cbox;

   double                min_dxyz;
   int                   cdir;
   int                   d, l;
                       
   int                   ierr = 0;
#if DEBUG
   char                  filename[255];
#endif

   /*-----------------------------------------------------
    * Refs to A,x,b (the PMatrix & PVectors within
    * the input SStructMatrix & SStructVectors)
    *-----------------------------------------------------*/
   hypre_SStructPMatrixRef(hypre_SStructMatrixPMatrix(A_in, 0), &A);
   hypre_SStructPVectorRef(hypre_SStructVectorPVector(b_in, 0), &b);
   hypre_SStructPVectorRef(hypre_SStructVectorPVector(x_in, 0), &x);

   /*-----------------------------------------------------
    * Set up coarse grids
    *-----------------------------------------------------*/

   grid  = hypre_SStructPMatrixPGrid(A);
   sgrid = hypre_SStructPGridSGrid(grid, 0);
   dim   = hypre_StructGridDim(sgrid);

   /* Compute a new max_levels value based on the grid */
   cbox = hypre_BoxDuplicate(hypre_StructGridBoundingBox(sgrid));
   max_levels =
      hypre_Log2(hypre_BoxSizeD(cbox, 0)) + 2 +
      hypre_Log2(hypre_BoxSizeD(cbox, 1)) + 2 +
      hypre_Log2(hypre_BoxSizeD(cbox, 2)) + 2;
   if ((sys_pfmg_data -> max_levels) > 0)
   {
      max_levels = hypre_min(max_levels, (sys_pfmg_data -> max_levels));
   }
   (sys_pfmg_data -> max_levels) = max_levels;

   /* compute dxyz */
   if ((dxyz[0] == 0) || (dxyz[1] == 0) || (dxyz[2] == 0))
   {
      hypre_PFMGComputeDxyz(hypre_SStructPMatrixSMatrix(A,0,0), dxyz);
   }

   grid_l = hypre_TAlloc(hypre_SStructPGrid *, max_levels);
   grid_l[0] = grid;
   P_grid_l = hypre_TAlloc(hypre_SStructPGrid *, max_levels);
   P_grid_l[0] = NULL;
   cdir_l = hypre_TAlloc(int, max_levels);
   active_l = hypre_TAlloc(int, max_levels);
   hypre_SetIndex(coarsen, 1, 1, 1); /* forces relaxation on finest grid */
   for (l = 0; ; l++)
   {
      /* determine cdir */
      min_dxyz = dxyz[0] + dxyz[1] + dxyz[2] + 1;
      cdir = -1;
      for (d = 0; d < dim; d++)
      {
         if ((hypre_BoxIMaxD(cbox, d) > hypre_BoxIMinD(cbox, d)) &&
             (dxyz[d] < min_dxyz))
         {
            min_dxyz = dxyz[d];
            cdir = d;
         }
      }

      /* if cannot coarsen in any direction, stop */
      if ( (cdir == -1) || (l == (max_levels - 1)) )
      {
         /* stop coarsening */
         active_l[l] = 1; /* forces relaxation on coarsest grid */
         break;
      }

      cdir_l[l] = cdir;

      if (hypre_IndexD(coarsen, cdir) != 0)
      {
         /* coarsened previously in this direction, relax level l */
         active_l[l] = 1;
         hypre_SetIndex(coarsen, 0, 0, 0);
         hypre_IndexD(coarsen, cdir) = 1;
      }
      else
      {
         active_l[l] = 0;
         hypre_IndexD(coarsen, cdir) = 1;
      }

      /* set cindex, findex, and stride */
      hypre_PFMGSetCIndex(cdir, cindex);
      hypre_PFMGSetFIndex(cdir, findex);
      hypre_PFMGSetStride(cdir, stride);

      /* update dxyz and coarsen cbox*/
      dxyz[cdir] *= 2;
      hypre_ProjectBox(cbox, cindex, stride);
      hypre_StructMapFineToCoarse(hypre_BoxIMin(cbox), cindex, stride,
                                  hypre_BoxIMin(cbox));
      hypre_StructMapFineToCoarse(hypre_BoxIMax(cbox), cindex, stride,
                                  hypre_BoxIMax(cbox));

      /* build the interpolation grid */
      hypre_SysStructCoarsen(grid_l[l], findex, stride, 0, &P_grid_l[l+1]);

      /* build the coarse grid */
      hypre_SysStructCoarsen(grid_l[l], cindex, stride, 1, &grid_l[l+1]);
   }
   num_levels = l + 1;
  
   /*-----------------------------------------------------
    * For fully periodic problems, the coarsest grid
    * problem (a single node) can have zero diagonal
    * blocks. This causes problems with the gselim
    * routine (which doesn't do pivoting). We avoid
    * this by skipping relaxation.
    *-----------------------------------------------------*/

   full_periodic = 1;
   for (d = 0; d < dim; d++)
   {
      full_periodic *= hypre_IndexD(hypre_SStructPGridPeriodic(grid),d);
   }
   if( full_periodic != 0)
   {
      hypre_SStructPGridDestroy(grid_l[num_levels-1]);
      hypre_SStructPGridDestroy(P_grid_l[num_levels-1]);
      num_levels -= 1;
   }

   /* free up some things */
   hypre_BoxDestroy(cbox);

   /* set all levels active if skip_relax = 0 */
   if (!skip_relax)
   {
      for (l = 0; l < num_levels; l++)
      {
         active_l[l] = 1;
      }
   }

   (sys_pfmg_data -> num_levels) = num_levels;
   (sys_pfmg_data -> cdir_l)     = cdir_l;
   (sys_pfmg_data -> active_l)   = active_l;
   (sys_pfmg_data -> grid_l)     = grid_l;
   (sys_pfmg_data -> P_grid_l)   = P_grid_l;

   /*-----------------------------------------------------
    * Set up matrix and vector structures
    *-----------------------------------------------------*/

   A_l  = hypre_TAlloc(hypre_SStructPMatrix *, num_levels);
   P_l  = hypre_TAlloc(hypre_SStructPMatrix *, num_levels - 1);
   RT_l = hypre_TAlloc(hypre_SStructPMatrix *, num_levels - 1);
   b_l  = hypre_TAlloc(hypre_SStructPVector *, num_levels);
   x_l  = hypre_TAlloc(hypre_SStructPVector *, num_levels);
   tx_l = hypre_TAlloc(hypre_SStructPVector *, num_levels);
   r_l  = tx_l;
   e_l  = tx_l;

   hypre_SStructPMatrixRef(A, &A_l[0]);
   hypre_SStructPVectorRef(b, &b_l[0]);
   hypre_SStructPVectorRef(x, &x_l[0]);

   hypre_SStructPVectorCreate(comm, grid_l[0], &tx_l[0]);
   hypre_SStructPVectorInitialize(tx_l[0]);

   for (l = 0; l < (num_levels - 1); l++)
   {
      cdir = cdir_l[l];

      P_l[l]  = hypre_SysPFMGCreateInterpOp(A_l[l], P_grid_l[l+1], cdir);
      hypre_SStructPMatrixInitialize(P_l[l]);

      RT_l[l] = P_l[l];

      A_l[l+1] = hypre_SysPFMGCreateRAPOp(RT_l[l], A_l[l], P_l[l],
                                          grid_l[l+1], cdir);
      hypre_SStructPMatrixInitialize(A_l[l+1]);

      hypre_SStructPVectorCreate(comm, grid_l[l+1], &b_l[l+1]);
      hypre_SStructPVectorInitialize(b_l[l+1]);

      hypre_SStructPVectorCreate(comm, grid_l[l+1], &x_l[l+1]);
      hypre_SStructPVectorInitialize(x_l[l+1]);

      hypre_SStructPVectorCreate(comm, grid_l[l+1], &tx_l[l+1]);
      hypre_SStructPVectorInitialize(tx_l[l+1]);
   }

   hypre_SStructPVectorAssemble(tx_l[0]);
   for (l = 0; l < (num_levels - 1); l++)
   {
      hypre_SStructPVectorAssemble(b_l[l+1]);
      hypre_SStructPVectorAssemble(x_l[l+1]);
      hypre_SStructPVectorAssemble(tx_l[l+1]);
   }

   (sys_pfmg_data -> A_l)  = A_l;
   (sys_pfmg_data -> P_l)  = P_l;
   (sys_pfmg_data -> RT_l) = RT_l;
   (sys_pfmg_data -> b_l)  = b_l;
   (sys_pfmg_data -> x_l)  = x_l;
   (sys_pfmg_data -> tx_l) = tx_l;
   (sys_pfmg_data -> r_l)  = r_l;
   (sys_pfmg_data -> e_l)  = e_l;

   /*-----------------------------------------------------
    * Set up multigrid operators and call setup routines
    *-----------------------------------------------------*/

   relax_data_l    = hypre_TAlloc(void *, num_levels);
   matvec_data_l   = hypre_TAlloc(void *, num_levels);
   restrict_data_l = hypre_TAlloc(void *, num_levels);
   interp_data_l   = hypre_TAlloc(void *, num_levels);

   for (l = 0; l < (num_levels - 1); l++)
   {
      cdir = cdir_l[l];

      hypre_PFMGSetCIndex(cdir, cindex);
      hypre_PFMGSetFIndex(cdir, findex);
      hypre_PFMGSetStride(cdir, stride);

      /* set up interpolation operator */
      hypre_SysPFMGSetupInterpOp(A_l[l], cdir, findex, stride, P_l[l]);

      /* set up the coarse grid operator */
      hypre_SysPFMGSetupRAPOp(RT_l[l], A_l[l], P_l[l],
                              cdir, cindex, stride, A_l[l+1]);

      /* set up the interpolation routine */
      hypre_SysSemiInterpCreate(&interp_data_l[l]);
      hypre_SysSemiInterpSetup(interp_data_l[l], P_l[l], 0, x_l[l+1], e_l[l],
                            cindex, findex, stride);

      /* set up the restriction routine */
      hypre_SysSemiRestrictCreate(&restrict_data_l[l]);
      hypre_SysSemiRestrictSetup(restrict_data_l[l], RT_l[l], 1, r_l[l], b_l[l+1],
                              cindex, findex, stride);
   }

   /* set up fine grid relaxation */
   relax_data_l[0] = hypre_SysPFMGRelaxCreate(comm);
   hypre_SysPFMGRelaxSetTol(relax_data_l[0], 0.0);
   hypre_SysPFMGRelaxSetType(relax_data_l[0], relax_type);
   hypre_SysPFMGRelaxSetTempVec(relax_data_l[0], tx_l[0]);
   hypre_SysPFMGRelaxSetup(relax_data_l[0], A_l[0], b_l[0], x_l[0]);
   if (num_levels > 1)
   {
      for (l = 1; l < (num_levels - 1); l++)
      {
         /* set up relaxation */
         relax_data_l[l] = hypre_SysPFMGRelaxCreate(comm);
         hypre_SysPFMGRelaxSetTol(relax_data_l[l], 0.0);
         hypre_SysPFMGRelaxSetType(relax_data_l[l], relax_type);
         hypre_SysPFMGRelaxSetTempVec(relax_data_l[l], tx_l[l]);
         hypre_SysPFMGRelaxSetup(relax_data_l[l], A_l[l], b_l[l], x_l[l]);
      }
      /* set up coarsest grid relaxation */
      relax_data_l[l] = hypre_SysPFMGRelaxCreate(comm);
      hypre_SysPFMGRelaxSetTol(relax_data_l[l], 0.0);
      hypre_SysPFMGRelaxSetMaxIter(relax_data_l[l], 1);
      hypre_SysPFMGRelaxSetType(relax_data_l[l], 0);
      hypre_SysPFMGRelaxSetTempVec(relax_data_l[l], tx_l[l]);
      hypre_SysPFMGRelaxSetup(relax_data_l[l], A_l[l], b_l[l], x_l[l]);
   }

   for (l = 0; l < num_levels; l++)
   {
      /* set up the residual routine */
      hypre_SStructPMatvecCreate(&matvec_data_l[l]);
      hypre_SStructPMatvecSetup(matvec_data_l[l], A_l[l], x_l[l]);
   }

   (sys_pfmg_data -> relax_data_l)    = relax_data_l;
   (sys_pfmg_data -> matvec_data_l)   = matvec_data_l;
   (sys_pfmg_data -> restrict_data_l) = restrict_data_l;
   (sys_pfmg_data -> interp_data_l)   = interp_data_l;

   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/

   if ((sys_pfmg_data -> logging) > 0)
   {
      max_iter = (sys_pfmg_data -> max_iter);
      (sys_pfmg_data -> norms)     = hypre_TAlloc(double, max_iter);
      (sys_pfmg_data -> rel_norms) = hypre_TAlloc(double, max_iter);
   }

#if DEBUG
   for (l = 0; l < (num_levels - 1); l++)
   {
      sprintf(filename, "zout_A.%02d", l);
      hypre_SStructPMatrixPrint(filename, A_l[l], 0);
      sprintf(filename, "zout_P.%02d", l);
      hypre_SStructPMatrixPrint(filename, P_l[l], 0);
   }
   sprintf(filename, "zout_A.%02d", l);
   hypre_SStructPMatrixPrint(filename, A_l[l], 0);
#endif

   /*-----------------------------------------------------
    * Destroy Refs to A,x,b (the PMatrix & PVectors within
    * the input SStructMatrix & SStructVectors).
    *-----------------------------------------------------*/
   hypre_SStructPMatrixDestroy(A);
   hypre_SStructPVectorDestroy(x);
   hypre_SStructPVectorDestroy(b);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SysStructCoarsen
 *--------------------------------------------------------------------------*/

int
hypre_SysStructCoarsen( hypre_SStructPGrid  *fgrid, 
                        hypre_Index          index,
                        hypre_Index          stride,
                        int                  prune,
                        hypre_SStructPGrid **cgrid_ptr )
{
   int ierr = 0;

   hypre_SStructPGrid   *cgrid;

   hypre_StructGrid     *sfgrid;
   hypre_StructGrid     *scgrid;

   MPI_Comm               comm;
   int                    ndim;
   int                    nvars;
   hypre_SStructVariable *vartypes;
   hypre_SStructVariable *new_vartypes;
   int                    i;
   int                    t;

   /*-----------------------------------------
    * Copy information from fine grid
    *-----------------------------------------*/

   comm      = hypre_SStructPGridComm(fgrid);
   ndim      = hypre_SStructPGridNDim(fgrid);
   nvars     = hypre_SStructPGridNVars(fgrid);
   vartypes  = hypre_SStructPGridVarTypes(fgrid);

   cgrid = hypre_TAlloc(hypre_SStructPGrid, 1);

   hypre_SStructPGridComm(cgrid)     = comm;
   hypre_SStructPGridNDim(cgrid)     = ndim;
   hypre_SStructPGridNVars(cgrid)    = nvars;
   new_vartypes = hypre_TAlloc(hypre_SStructVariable, nvars);
   for (i = 0; i < nvars; i++)
   {
      new_vartypes[i] = vartypes[i];
   }
   hypre_SStructPGridVarTypes(cgrid) = new_vartypes;

   for (t = 0; t < 8; t++)
   {
      hypre_SStructPGridVTSGrid(cgrid, t)     = NULL;
      hypre_SStructPGridVTIBoxArray(cgrid, t) = NULL;
   }

   /*-----------------------------------------
    * Set the coarse sgrid
    *-----------------------------------------*/

   sfgrid = hypre_SStructPGridCellSGrid(fgrid);
   hypre_StructCoarsen(sfgrid, index, stride, prune, &scgrid); 

   hypre_CopyIndex(hypre_StructGridPeriodic(scgrid),
                   hypre_SStructPGridPeriodic(cgrid));

   hypre_SStructPGridCellSGrid(cgrid) = scgrid;

   hypre_SStructPGridLocalSize(cgrid)  = 0;
   hypre_SStructPGridGlobalSize(cgrid) = 0;

   hypre_SStructPGridAssemble(cgrid);

   *cgrid_ptr = cgrid;

   return ierr;
}

