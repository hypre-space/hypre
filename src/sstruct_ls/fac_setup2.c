/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.15 $
 ***********************************************************************EHEADER*/

#include "headers.h"
#include "fac.h"

/*--------------------------------------------------------------------------
 * hypre_FacSetup2: Constructs the level composite structures.
 * Each consists only of two levels, the refinement patches and the
 * coarse parent base grids.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FacSetup2( void                 *fac_vdata,
                hypre_SStructMatrix  *A_in,
                hypre_SStructVector  *b,
                hypre_SStructVector  *x )
{
   hypre_FACData          *fac_data      =  fac_vdata;

   HYPRE_Int              *plevels       = (fac_data-> plevels);
   hypre_Index            *rfactors      = (fac_data-> prefinements);

   MPI_Comm                comm;
   HYPRE_Int               ndim;
   HYPRE_Int               npart;
   HYPRE_Int               nparts_level  =  2;
   HYPRE_Int               part_crse     =  0;
   HYPRE_Int               part_fine     =  1;
   hypre_SStructPMatrix   *A_pmatrix;
   hypre_StructMatrix     *A_smatrix;
   hypre_Box              *A_smatrix_dbox;

   hypre_SStructGrid     **grid_level;
   hypre_SStructGraph    **graph_level;
   HYPRE_Int               part, level;
   HYPRE_Int               nvars;

   hypre_SStructGraph     *graph;
   hypre_SStructGrid      *grid;
   hypre_SStructPGrid     *pgrid; 
   hypre_StructGrid       *sgrid; 
   hypre_BoxArray         *sgrid_boxes;
   hypre_Box              *sgrid_box;
   hypre_SStructStencil   *stencils;
   hypre_BoxArray         *iboxarray;

   hypre_Index            *refine_factors;
   hypre_IndexRef          box_start;
   hypre_IndexRef          box_end;

   hypre_SStructUVEntry  **Uventries;
   HYPRE_Int               nUventries;
   HYPRE_Int              *iUventries;
   hypre_SStructUVEntry   *Uventry; 
   hypre_SStructUEntry    *Uentry;
   hypre_Index             index, to_index, stride;
   HYPRE_Int               var, to_var, to_part, level_part, level_topart;
   HYPRE_Int               var1, var2;
   HYPRE_Int               i, j, k, to_rank, row_coord, nUentries;
   hypre_BoxManEntry      *boxman_entry;

   hypre_SStructMatrix    *A_rap;
   hypre_SStructMatrix   **A_level;
   hypre_SStructVector   **b_level;
   hypre_SStructVector   **x_level;
   hypre_SStructVector   **r_level;
   hypre_SStructVector   **e_level;
   hypre_SStructPVector  **tx_level;
   hypre_SStructVector    *tx;

   void                  **matvec_data_level;
   void                  **pmatvec_data_level;
   void                   *matvec_data;
   void                  **relax_data_level;
   void                  **interp_data_level;
   void                  **restrict_data_level;


   /* coarsest grid solver */
   HYPRE_Int               csolver_type       =(fac_data-> csolver_type);
   HYPRE_SStructSolver     crse_solver;
   HYPRE_SStructSolver     crse_precond;
   
   HYPRE_Int               max_level        =  hypre_FACDataMaxLevels(fac_data);
   HYPRE_Int               relax_type       =  fac_data -> relax_type;
   HYPRE_Int               usr_jacobi_weight=  fac_data -> usr_jacobi_weight;
   double                  jacobi_weight    =  fac_data -> jacobi_weight;
   HYPRE_Int              *levels;
   HYPRE_Int              *part_to_level;

   HYPRE_Int               box, box_volume;
   HYPRE_Int               max_box_volume;
   HYPRE_Int               stencil_size;
   hypre_Index             stencil_shape_i, loop_size;
   HYPRE_Int              *stencil_vars;
   double                 *values;
   double                 *A_smatrix_value;
   HYPRE_Int               iA, loopi, loopj, loopk;
 
   HYPRE_Int              *nrows;
   HYPRE_Int             **ncols;
   HYPRE_Int             **rows;
   HYPRE_Int             **cols;
   HYPRE_Int              *cnt;
   double                 *vals;
   
   HYPRE_Int              *level_rows;
   HYPRE_Int              *level_cols;
   HYPRE_Int               level_cnt;

   HYPRE_IJMatrix          ij_A;
   HYPRE_Int               matrix_type;

   HYPRE_Int               max_cycles;

   HYPRE_Int               ierr = 0;
/*hypre_SStructMatrix *nested_A;

   nested_A= hypre_TAlloc(hypre_SStructMatrix , 1);
   nested_A= hypre_CoarsenAMROp(fac_vdata, A);*/

   /* generate the composite operator with the computed coarse-grid operators */
    hypre_AMR_RAP(A_in, rfactors, &A_rap);
   (fac_data -> A_rap)= A_rap;

    comm = hypre_SStructMatrixComm(A_rap);
    ndim = hypre_SStructMatrixNDim(A_rap);
    npart= hypre_SStructMatrixNParts(A_rap);
    graph= hypre_SStructMatrixGraph(A_rap);
    grid = hypre_SStructGraphGrid(graph);
    ij_A = hypre_SStructMatrixIJMatrix(A_rap);
    matrix_type= hypre_SStructMatrixObjectType(A_rap);

   /*--------------------------------------------------------------------------
    * logging arrays.
    *--------------------------------------------------------------------------*/
   if ((fac_data -> logging) > 0)
   {
      max_cycles = (fac_data -> max_cycles);
     (fac_data -> norms)    = hypre_TAlloc(double, max_cycles);
     (fac_data -> rel_norms)= hypre_TAlloc(double, max_cycles);
   }

   /*--------------------------------------------------------------------------
    * Extract the amr/sstruct level/part structure and refinement factors.
    *--------------------------------------------------------------------------*/
   levels        = hypre_CTAlloc(HYPRE_Int, npart);
   part_to_level = hypre_CTAlloc(HYPRE_Int, npart);
   refine_factors= hypre_CTAlloc(hypre_Index, npart);
   for (part= 0; part< npart; part++)
   {
       part_to_level[part]  = plevels[part];
       levels[plevels[part]]= part;
       for (i= 0; i< ndim; i++)
       {
           refine_factors[plevels[part]][i]= rfactors[part][i];
       }
       for (i= ndim; i< 3; i++)
       {
           refine_factors[plevels[part]][i]= 1;
       }
   }
   (fac_data -> level_to_part) = levels;
   (fac_data -> part_to_level) = part_to_level;
   (fac_data -> refine_factors)= refine_factors;
   
   /*--------------------------------------------------------------------------
    * Create the level SStructGrids using the original composite grid. 
    *--------------------------------------------------------------------------*/
   grid_level= hypre_TAlloc(hypre_SStructGrid *, max_level+1);
   for (level= max_level; level >= 0; level--)
   {
       HYPRE_SStructGridCreate(comm, ndim, nparts_level, &grid_level[level]);
   }

   for (level= max_level; level >= 0; level--)
   {
       /*--------------------------------------------------------------------------
        * Create the fine part of the finest level SStructGrids using the original 
        * composite grid.
        *--------------------------------------------------------------------------*/
       if (level == max_level)
       {
          pgrid = hypre_SStructGridPGrid(grid, levels[level]);
          iboxarray= hypre_SStructPGridCellIBoxArray(pgrid);
          for (box = 0; box < hypre_BoxArraySize(iboxarray); box++)
          {
              HYPRE_SStructGridSetExtents(grid_level[level], part_fine,
                                          hypre_BoxIMin( hypre_BoxArrayBox(iboxarray,box) ),
                                          hypre_BoxIMax( hypre_BoxArrayBox(iboxarray,box) ));
          }

          HYPRE_SStructGridSetVariables( grid_level[level], part_fine, 
                                         hypre_SStructPGridNVars(pgrid), 
                                         hypre_SStructPGridVarTypes(pgrid) );

          /*-----------------------------------------------------------------------
           * Create the coarsest level grid if A has only 1 level
           *-----------------------------------------------------------------------*/
          if (level == 0)
          {
             for (box = 0; box < hypre_BoxArraySize(iboxarray); box++)
             {
                HYPRE_SStructGridSetExtents(grid_level[level], part_crse,
                                          hypre_BoxIMin( hypre_BoxArrayBox(iboxarray,box) ),
                                          hypre_BoxIMax( hypre_BoxArrayBox(iboxarray,box) ));
             }

             HYPRE_SStructGridSetVariables( grid_level[level], part_crse,
                                            hypre_SStructPGridNVars(pgrid),
                                            hypre_SStructPGridVarTypes(pgrid) );
          }
       }

       /*--------------------------------------------------------------------------
        * Create the coarse part of level SStructGrids using the original composite 
        * grid, the coarsest part SStructGrid, and the fine part if level < max_level.
        *--------------------------------------------------------------------------*/
       if (level > 0)
       {
          pgrid = hypre_SStructGridPGrid(grid, levels[level-1]);
          iboxarray= hypre_SStructPGridCellIBoxArray(pgrid);
          for (box = 0; box < hypre_BoxArraySize(iboxarray); box++)
          {
              HYPRE_SStructGridSetExtents(grid_level[level], part_crse,
                                          hypre_BoxIMin( hypre_BoxArrayBox(iboxarray,box) ),
                                          hypre_BoxIMax( hypre_BoxArrayBox(iboxarray,box) ));

              HYPRE_SStructGridSetExtents(grid_level[level-1], part_fine,
                                          hypre_BoxIMin( hypre_BoxArrayBox(iboxarray,box) ),
                                          hypre_BoxIMax( hypre_BoxArrayBox(iboxarray,box) ));

              
              if (level == 1)
              {
                  HYPRE_SStructGridSetExtents(grid_level[level-1], part_crse,
                                              hypre_BoxIMin( hypre_BoxArrayBox(iboxarray,box) ),
                                              hypre_BoxIMax( hypre_BoxArrayBox(iboxarray,box) ));
              }
          }

          HYPRE_SStructGridSetVariables( grid_level[level], part_crse, 
                                         hypre_SStructPGridNVars(pgrid), 
                                         hypre_SStructPGridVarTypes(pgrid) );

          HYPRE_SStructGridSetVariables( grid_level[level-1], part_fine, 
                                         hypre_SStructPGridNVars(pgrid), 
                                         hypre_SStructPGridVarTypes(pgrid) );

          /* coarsest SStructGrid */
          if (level == 1)
          {
             HYPRE_SStructGridSetVariables( grid_level[level-1], part_crse, 
                                            hypre_SStructPGridNVars(pgrid), 
                                            hypre_SStructPGridVarTypes(pgrid) );
          }
       }

       HYPRE_SStructGridAssemble(grid_level[level]);
   }

   (fac_data -> grid_level)= grid_level;

   /*-----------------------------------------------------------
    * Set up the graph. Create only the structured components
    * first.
    *-----------------------------------------------------------*/
   graph_level= hypre_TAlloc(hypre_SStructGraph *, max_level+1);
   for (level= max_level; level >= 0; level--)
   {
       HYPRE_SStructGraphCreate(comm, grid_level[level], &graph_level[level]);
   }

   for (level= max_level; level >= 0; level--)
   {
       /*-----------------------------------------------------------------------
        * Create the fine part of the finest level structured graph connection.
        *-----------------------------------------------------------------------*/
       if (level == max_level)
       {
           pgrid = hypre_SStructGridPGrid(grid, levels[level]);
           nvars = hypre_SStructPGridNVars(pgrid);
           for (var1 = 0; var1 < nvars; var1++)
           {
              stencils= hypre_SStructGraphStencil(graph, levels[level], var1);
              HYPRE_SStructGraphSetStencil(graph_level[level], part_fine, var1, stencils);

              if (level == 0)
              {
                 HYPRE_SStructGraphSetStencil(graph_level[level], part_crse, var1, stencils);
              }
           }
       }

       /*--------------------------------------------------------------------------
        * Create the coarse part of the graph_level using the graph of A, and the
        * and the fine part if level < max_level.
        *--------------------------------------------------------------------------*/
       if (level > 0)
       {
           pgrid = hypre_SStructGridPGrid(grid, levels[level-1]);
           nvars = hypre_SStructPGridNVars(pgrid);

           for (var1 = 0; var1 < nvars; var1++)
           {
              stencils= hypre_SStructGraphStencil(graph, levels[level-1], var1);
              HYPRE_SStructGraphSetStencil(graph_level[level], part_crse, var1, stencils );
              HYPRE_SStructGraphSetStencil(graph_level[level-1], part_fine, var1, stencils );

              if (level == 1)
              {
                 HYPRE_SStructGraphSetStencil(graph_level[level-1], part_crse, var1, stencils );
              }

           }
       }
   }

   /*-----------------------------------------------------------
    * Extract the non-stencil graph structure: assuming only like
    * variables connect. Also count the number of unstructured
    * connections per part.
    *
    * THE COARSEST COMPOSITE MATRIX DOES NOT HAVE ANY NON-STENCIL
    * CONNECTIONS.
    *-----------------------------------------------------------*/
   Uventries =  hypre_SStructGraphUVEntries(graph);
   nUventries=  hypre_SStructGraphNUVEntries(graph);
   iUventries=  hypre_SStructGraphIUVEntries(graph);

   nrows     =  hypre_CTAlloc(HYPRE_Int, max_level+1);
   for (i= 0; i< nUventries; i++)
   {
      Uventry=  Uventries[iUventries[i]];

      part     =  hypre_SStructUVEntryPart(Uventry);
      hypre_CopyIndex(hypre_SStructUVEntryIndex(Uventry), index);
      var      =  hypre_SStructUVEntryVar(Uventry);
      nUentries=  hypre_SStructUVEntryNUEntries(Uventry);

      for (k= 0; k< nUentries; k++)
      {
         Uentry  =  hypre_SStructUVEntryUEntry(Uventry, k);
    
         to_part =  hypre_SStructUEntryToPart(Uentry);
         hypre_CopyIndex(hypre_SStructUEntryToIndex(Uentry), to_index);
         to_var  =  hypre_SStructUEntryToVar(Uentry);

         if ( part_to_level[part] >= part_to_level[to_part] )
         {
            level        = part_to_level[part];
            level_part   = part_fine;
            level_topart = part_crse;
         }
         else
         {
            level        = part_to_level[to_part];
            level_part   = part_crse;
            level_topart = part_fine;
         }
         nrows[level]++;

         HYPRE_SStructGraphAddEntries(graph_level[level], level_part, index,
                                      var, level_topart, to_index, to_var);
      }
   }

   for (level= 0; level <= max_level; level++)
   {
       HYPRE_SStructGraphAssemble(graph_level[level]);
   }

   (fac_data -> graph_level)= graph_level;

   /*---------------------------------------------------------------
    * Create the level SStruct_Vectors, and temporary global
    * sstuct_vector. 
    *---------------------------------------------------------------*/
   b_level= hypre_TAlloc(hypre_SStructVector *, max_level+1);
   x_level= hypre_TAlloc(hypre_SStructVector *, max_level+1);
   r_level= hypre_TAlloc(hypre_SStructVector *, max_level+1);
   e_level= hypre_TAlloc(hypre_SStructVector *, max_level+1);

   tx_level= hypre_TAlloc(hypre_SStructPVector *, max_level+1);

   for (level= 0; level<= max_level; level++)
   {
       HYPRE_SStructVectorCreate(comm, grid_level[level], &b_level[level]);
       HYPRE_SStructVectorInitialize(b_level[level]);
       HYPRE_SStructVectorAssemble(b_level[level]);

       HYPRE_SStructVectorCreate(comm, grid_level[level], &x_level[level]);
       HYPRE_SStructVectorInitialize(x_level[level]);
       HYPRE_SStructVectorAssemble(x_level[level]);

       HYPRE_SStructVectorCreate(comm, grid_level[level], &r_level[level]);
       HYPRE_SStructVectorInitialize(r_level[level]);
       HYPRE_SStructVectorAssemble(r_level[level]);

       HYPRE_SStructVectorCreate(comm, grid_level[level], &e_level[level]);
       HYPRE_SStructVectorInitialize(e_level[level]);
       HYPRE_SStructVectorAssemble(e_level[level]);

       /* temporary vector for fine patch relaxation */
       hypre_SStructPVectorCreate(comm,
                                  hypre_SStructGridPGrid(grid_level[level], part_fine),
                                  &tx_level[level]);
       hypre_SStructPVectorInitialize(tx_level[level]);
       hypre_SStructPVectorAssemble(tx_level[level]);

   }

   /* temp SStructVectors */
   HYPRE_SStructVectorCreate(comm, grid, &tx);
   HYPRE_SStructVectorInitialize(tx);
   HYPRE_SStructVectorAssemble(tx);

   (fac_data -> b_level) = b_level;
   (fac_data -> x_level) = x_level;
   (fac_data -> r_level) = r_level;
   (fac_data -> e_level) = e_level;
   (fac_data -> tx_level)= tx_level;
   (fac_data -> tx)      = tx;

   /*-----------------------------------------------------------
    * Set up the level composite sstruct_matrices. 
    *-----------------------------------------------------------*/

   A_level= hypre_TAlloc(hypre_SStructMatrix *, max_level+1);
   hypre_SetIndex(stride, 1, 1, 1);
   for (level= 0; level <= max_level; level++)
   {
       HYPRE_SStructMatrixCreate(comm, graph_level[level], &A_level[level]);
       HYPRE_SStructMatrixInitialize(A_level[level]);

       max_box_volume= 0;
       pgrid = hypre_SStructGridPGrid(grid, levels[level]);
       nvars = hypre_SStructPGridNVars(pgrid);

       for (var1 = 0; var1 < nvars; var1++)
       {
          sgrid= hypre_SStructPGridSGrid(pgrid, var1);
          sgrid_boxes= hypre_StructGridBoxes(sgrid);

          hypre_ForBoxI(i, sgrid_boxes)
          {
             sgrid_box = hypre_BoxArrayBox(sgrid_boxes, i);
             box_volume= hypre_BoxVolume(sgrid_box);

             max_box_volume= hypre_max(max_box_volume, box_volume);
          }
       }

       values   = hypre_TAlloc(double, max_box_volume);
       A_pmatrix= hypre_SStructMatrixPMatrix(A_rap, levels[level]);

       /*-----------------------------------------------------------
        * extract stencil values for all fine levels.
        *-----------------------------------------------------------*/
       for (var1 = 0; var1 < nvars; var1++)
       {
          sgrid= hypre_SStructPGridSGrid(pgrid, var1);
          sgrid_boxes= hypre_StructGridBoxes(sgrid);

          stencils= hypre_SStructGraphStencil(graph, levels[level], var1);
          stencil_size= hypre_SStructStencilSize(stencils);
          stencil_vars= hypre_SStructStencilVars(stencils);

          for (i = 0; i < stencil_size; i++)
          {
             var2= stencil_vars[i];
             A_smatrix= hypre_SStructPMatrixSMatrix(A_pmatrix, var1, var2);
             hypre_CopyIndex(hypre_SStructStencilEntry(stencils, i), stencil_shape_i);

             hypre_ForBoxI(j, sgrid_boxes)
             {
                sgrid_box=  hypre_BoxArrayBox(sgrid_boxes, j);
                box_start=  hypre_BoxIMin(sgrid_box);
                box_end  =  hypre_BoxIMax(sgrid_box);

                A_smatrix_dbox=  hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A_smatrix), j);
                A_smatrix_value= 
                      hypre_StructMatrixExtractPointerByIndex(A_smatrix, j, stencil_shape_i);

                hypre_BoxGetSize(sgrid_box, loop_size);

                hypre_BoxLoop2Begin(loop_size,
                                    sgrid_box, box_start, stride, k,
                                    A_smatrix_dbox, box_start, stride, iA);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,k,iA
#include "hypre_box_smp_forloop.h"
                hypre_BoxLoop2For(loopi, loopj, loopk, k, iA)
                {
                   values[k]= A_smatrix_value[iA];
                }
                hypre_BoxLoop2End(k, iA);

                HYPRE_SStructMatrixSetBoxValues(A_level[level], part_fine, box_start, box_end,
                                                var1, 1, &i, values);
             }   /* hypre_ForBoxI */ 
          }      /* for i */
       }         /* for var1 */
       hypre_TFree(values);

      /*-----------------------------------------------------------
       *  Extract the coarse part 
       *-----------------------------------------------------------*/
      if (level > 0)
      {
          max_box_volume= 0;
          pgrid = hypre_SStructGridPGrid(grid, levels[level-1]);
          nvars = hypre_SStructPGridNVars(pgrid);

          for (var1 = 0; var1 < nvars; var1++)
          {
             sgrid      = hypre_SStructPGridSGrid( pgrid, var1 );
             sgrid_boxes= hypre_StructGridBoxes(sgrid);

             hypre_ForBoxI( i, sgrid_boxes )
             {
                sgrid_box = hypre_BoxArrayBox(sgrid_boxes, i);
                box_volume= hypre_BoxVolume(sgrid_box);

                max_box_volume= hypre_max(max_box_volume, box_volume );
             }
          }

          values   = hypre_TAlloc(double, max_box_volume);
          A_pmatrix= hypre_SStructMatrixPMatrix(A_rap, levels[level-1]);

          /*-----------------------------------------------------------
           * extract stencil values 
           *-----------------------------------------------------------*/
          for (var1 = 0; var1 < nvars; var1++)
          {
             sgrid      = hypre_SStructPGridSGrid(pgrid, var1);
             sgrid_boxes= hypre_StructGridBoxes(sgrid);

             stencils= hypre_SStructGraphStencil(graph, levels[level-1], var1);
             stencil_size= hypre_SStructStencilSize(stencils);
             stencil_vars= hypre_SStructStencilVars(stencils);

             for (i = 0; i < stencil_size; i++)
             {
                var2= stencil_vars[i];
                A_smatrix= hypre_SStructPMatrixSMatrix(A_pmatrix, var1, var2);
                hypre_CopyIndex(hypre_SStructStencilEntry(stencils, i), stencil_shape_i);

                hypre_ForBoxI( j, sgrid_boxes )
                {
                   sgrid_box=  hypre_BoxArrayBox(sgrid_boxes, j);
                   box_start=  hypre_BoxIMin(sgrid_box);
                   box_end  =  hypre_BoxIMax(sgrid_box);

                   A_smatrix_dbox=  hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A_smatrix), j);
                   A_smatrix_value= 
                        hypre_StructMatrixExtractPointerByIndex(A_smatrix, j, stencil_shape_i);

                   hypre_BoxGetSize(sgrid_box, loop_size);

                   hypre_BoxLoop2Begin(loop_size,
                                       sgrid_box, box_start, stride, k,
                                       A_smatrix_dbox, box_start, stride, iA);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,k,iA
#include "hypre_box_smp_forloop.h"
                   hypre_BoxLoop2For(loopi, loopj, loopk, k, iA)
                   {
                       values[k]= A_smatrix_value[iA];
                   }
                   hypre_BoxLoop2End(k, iA);

                   HYPRE_SStructMatrixSetBoxValues(A_level[level], part_crse, box_start, box_end,
                                                   var1, 1, &i, values);
                }  /* hypre_ForBoxI */ 
             }     /* for i */
          }        /* for var1 */
          hypre_TFree(values);
      }            /* if level > 0 */
   }               /* for level */

   /*-----------------------------------------------------------
    * extract the non-stencil values for all but the coarsest
    * level sstruct_matrix. Use the HYPRE_IJMatrixGetValues
    * for each level of A.
    *-----------------------------------------------------------*/

   Uventries =  hypre_SStructGraphUVEntries(graph);
   nUventries=  hypre_SStructGraphNUVEntries(graph);
   iUventries=  hypre_SStructGraphIUVEntries(graph);

   /*-----------------------------------------------------------
    * Allocate memory for arguments of HYPRE_IJMatrixGetValues.
    *-----------------------------------------------------------*/
   ncols =  hypre_TAlloc(HYPRE_Int *, max_level+1);
   rows  =  hypre_TAlloc(HYPRE_Int *, max_level+1);
   cols  =  hypre_TAlloc(HYPRE_Int *, max_level+1);
   cnt   =  hypre_CTAlloc(HYPRE_Int, max_level+1);

   ncols[0]= NULL;
   rows[0] = NULL;
   cols[0] = NULL;
   for (level= 1; level<= max_level; level++)
   {
      ncols[level]= hypre_TAlloc(HYPRE_Int, nrows[level]);
      for (i=0; i< nrows[level]; i++)
      {
         ncols[level][i]= 1;
      }
      rows[level] = hypre_TAlloc(HYPRE_Int, nrows[level]);
      cols[level] = hypre_TAlloc(HYPRE_Int, nrows[level]);
   }
   
   for (i= 0; i< nUventries; i++)
   {
      Uventry  =  Uventries[iUventries[i]];

      part     =  hypre_SStructUVEntryPart(Uventry);
      hypre_CopyIndex(hypre_SStructUVEntryIndex(Uventry), index);
      var      =  hypre_SStructUVEntryVar(Uventry);

      hypre_SStructGridFindBoxManEntry(grid, part, index, var, &boxman_entry);
      hypre_SStructBoxManEntryGetGlobalRank(boxman_entry, index, &row_coord,
                                         matrix_type);

      nUentries=  hypre_SStructUVEntryNUEntries(Uventry);
      for (k= 0; k< nUentries; k++)
      {
         to_part =  hypre_SStructUVEntryToPart(Uventry, k);
         to_rank =  hypre_SStructUVEntryRank(Uventry, k);
         
         /*-----------------------------------------------------------
          *  store the row & col indices in the correct level.
          *-----------------------------------------------------------*/
         level   = hypre_max( part_to_level[part], part_to_level[to_part] );
         rows[level][ cnt[level] ]= row_coord;
         cols[level][ cnt[level]++ ]= to_rank;
      }
   }
   hypre_TFree(cnt);

   for (level= 1; level<= max_level; level++)
   {
  
      vals      = hypre_CTAlloc(double, nrows[level]);
      level_rows= hypre_TAlloc(HYPRE_Int, nrows[level]);
      level_cols= hypre_TAlloc(HYPRE_Int, nrows[level]);

      HYPRE_IJMatrixGetValues(ij_A, nrows[level], ncols[level], rows[level], 
                              cols[level], vals);

      Uventries =  hypre_SStructGraphUVEntries(graph_level[level]);
      /*-----------------------------------------------------------
       * Find the rows & cols of the level ij_matrices where the
       * extracted data must be placed. Note that because the
       * order in which the HYPRE_SStructGraphAddEntries in the
       * graph_level's is the same order in which rows[level] & 
       * cols[level] were formed, the coefficients in val are
       * in the correct order.
       *-----------------------------------------------------------*/

      level_cnt= 0;
      for (i= 0; i< hypre_SStructGraphNUVEntries(graph_level[level]); i++)
      {
         j      =  hypre_SStructGraphIUVEntry(graph_level[level], i);
         Uventry=  Uventries[j];

         part     =  hypre_SStructUVEntryPart(Uventry);
         hypre_CopyIndex(hypre_SStructUVEntryIndex(Uventry), index);
         var      =  hypre_SStructUVEntryVar(Uventry);
   
         hypre_SStructGridFindBoxManEntry(grid_level[level], part, index, var, &boxman_entry);
         hypre_SStructBoxManEntryGetGlobalRank(boxman_entry, index, &row_coord, matrix_type);

         nUentries=  hypre_SStructUVEntryNUEntries(Uventry);
         for (k= 0; k< nUentries; k++)
         {
            to_rank =  hypre_SStructUVEntryRank(Uventry, k);
         
            level_rows[level_cnt]  = row_coord;
            level_cols[level_cnt++]= to_rank;
         }
      }
    
      /*-----------------------------------------------------------
       * Place the extracted ij coefficients into the level ij
       * matrices.
       *-----------------------------------------------------------*/
      HYPRE_IJMatrixSetValues( hypre_SStructMatrixIJMatrix(A_level[level]),
                               nrows[level], ncols[level], (const HYPRE_Int *) level_rows, 
                              (const HYPRE_Int *) level_cols, (const double *) vals );
      
      hypre_TFree(ncols[level]);
      hypre_TFree(rows[level]);
      hypre_TFree(cols[level]);

      hypre_TFree(vals);
      hypre_TFree(level_rows);
      hypre_TFree(level_cols);
   }

   hypre_TFree(ncols);
   hypre_TFree(rows);
   hypre_TFree(cols);
   hypre_TFree(nrows);
 
   /*---------------------------------------------------------------
    * Construct the fine grid (part 1) SStruct_PMatrix for all
    * levels except for max_level. This involves coarsening the
    * finer level SStruct_Matrix. Coarsening involves interpolation, 
    * matvec, and restriction (to obtain the "row-sum").
    *---------------------------------------------------------------*/
   matvec_data_level  = hypre_TAlloc(void *, max_level+1);
   pmatvec_data_level = hypre_TAlloc(void *, max_level+1);
   interp_data_level  = hypre_TAlloc(void *, max_level+1);
   restrict_data_level= hypre_TAlloc(void *, max_level+1);
   for (level= 0; level<= max_level; level++)
   {
       if (level < max_level)
       {
          hypre_FacSemiInterpCreate2(&interp_data_level[level]);
          hypre_FacSemiInterpSetup2(interp_data_level[level],
                                   x_level[level+1], 
                                   hypre_SStructVectorPVector(x_level[level], part_fine),
                                   refine_factors[level+1]);
       }
       else
       {
          interp_data_level[level]= NULL;
       }

       if (level > 0)
       {
          hypre_FacSemiRestrictCreate2(&restrict_data_level[level]);

          hypre_FacSemiRestrictSetup2(restrict_data_level[level],
                                     x_level[level], part_crse, part_fine,
                                     hypre_SStructVectorPVector(x_level[level-1], part_fine),
                                     refine_factors[level]);
       }
       else
       {
          restrict_data_level[level]= NULL;
       }
   }

   for (level= max_level; level> 0; level--)
   {
      
     /*  hypre_FacZeroCFSten(hypre_SStructMatrixPMatrix(A_level[level], part_fine),
                           hypre_SStructMatrixPMatrix(A_level[level], part_crse),
                           grid_level[level],
                           part_fine,
                           refine_factors[level]);
       hypre_FacZeroFCSten(hypre_SStructMatrixPMatrix(A_level[level], part_fine),
                           grid_level[level],
                           part_fine);
      */

       hypre_ZeroAMRMatrixData(A_level[level], part_crse, refine_factors[level]);


       HYPRE_SStructMatrixAssemble(A_level[level]);
       /*------------------------------------------------------------
        * create data structures that are needed for coarsening 
        -------------------------------------------------------------*/
       hypre_SStructMatvecCreate(&matvec_data_level[level]);
       hypre_SStructMatvecSetup(matvec_data_level[level],
                                A_level[level],
                                x_level[level]);

       hypre_SStructPMatvecCreate(&pmatvec_data_level[level]);
       hypre_SStructPMatvecSetup(pmatvec_data_level[level],
                                 hypre_SStructMatrixPMatrix(A_level[level],part_fine),
                                 hypre_SStructVectorPVector(x_level[level],part_fine));
   } 

   /*---------------------------------------------------------------
    * To avoid memory leaks, we cannot reference the coarsest level
    * SStructPMatrix. We need only copy the stuctured coefs.
    *---------------------------------------------------------------*/
   pgrid= hypre_SStructGridPGrid(grid_level[0], part_fine);
   nvars= hypre_SStructPGridNVars(pgrid);
   A_pmatrix= hypre_SStructMatrixPMatrix(A_level[0], part_fine);
   for (var1 = 0; var1 < nvars; var1++)
   {
      sgrid= hypre_SStructPGridSGrid(pgrid, var1);
      sgrid_boxes= hypre_StructGridBoxes(sgrid);

      max_box_volume= 0;
      hypre_ForBoxI(i, sgrid_boxes)
      {
          sgrid_box = hypre_BoxArrayBox(sgrid_boxes, i);
          box_volume= hypre_BoxVolume(sgrid_box);

          max_box_volume= hypre_max(max_box_volume, box_volume);
      }

      values   = hypre_TAlloc(double, max_box_volume);
   
      stencils= hypre_SStructGraphStencil(graph_level[0], part_fine, var1);
      stencil_size= hypre_SStructStencilSize(stencils);
      stencil_vars= hypre_SStructStencilVars(stencils);

      for (i = 0; i < stencil_size; i++)
      {
         var2= stencil_vars[i];
         A_smatrix= hypre_SStructPMatrixSMatrix(A_pmatrix, var1, var2);
         hypre_CopyIndex(hypre_SStructStencilEntry(stencils, i), stencil_shape_i);
         hypre_ForBoxI(j, sgrid_boxes)
         {
             sgrid_box=  hypre_BoxArrayBox(sgrid_boxes, j);
             box_start=  hypre_BoxIMin(sgrid_box);
             box_end  =  hypre_BoxIMax(sgrid_box);

             A_smatrix_dbox=  hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A_smatrix), j);
             A_smatrix_value=
                        hypre_StructMatrixExtractPointerByIndex(A_smatrix, j, stencil_shape_i);

             hypre_BoxGetSize(sgrid_box, loop_size);

             hypre_BoxLoop2Begin(loop_size,
                                 sgrid_box, box_start, stride, k,
                                 A_smatrix_dbox, box_start, stride, iA);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,k,iA
#include "hypre_box_smp_forloop.h"
             hypre_BoxLoop2For(loopi, loopj, loopk, k, iA)
             {
                 values[k]= A_smatrix_value[iA];
             }
             hypre_BoxLoop2End(k, iA);

             HYPRE_SStructMatrixSetBoxValues(A_level[0], part_crse, box_start, box_end,
                                             var1, 1, &i, values);
         }   /* hypre_ForBoxI */
      }      /* for i */
      
      hypre_TFree(values);
   }         /* for var1 */

   HYPRE_SStructMatrixAssemble(A_level[0]);

   hypre_SStructMatvecCreate(&matvec_data_level[0]);
   hypre_SStructMatvecSetup(matvec_data_level[0],
                            A_level[0],
                            x_level[0]);

   hypre_SStructPMatvecCreate(&pmatvec_data_level[0]);
   hypre_SStructPMatvecSetup(pmatvec_data_level[0],
                             hypre_SStructMatrixPMatrix(A_level[0],part_fine),
                             hypre_SStructVectorPVector(x_level[0],part_fine));
   
   hypre_SStructMatvecCreate(&matvec_data);
   hypre_SStructMatvecSetup(matvec_data, A_rap, x);

   /*HYPRE_SStructVectorPrint("sstruct.out.b_l", b_level[max_level], 0);*/
   /*HYPRE_SStructMatrixPrint("sstruct.out.A_l",  A_level[max_level-2], 0);*/
   (fac_data -> A_level)             = A_level;
   (fac_data -> matvec_data_level)   = matvec_data_level;
   (fac_data -> pmatvec_data_level)  = pmatvec_data_level;
   (fac_data -> matvec_data)         = matvec_data;
   (fac_data -> interp_data_level)   = interp_data_level;
   (fac_data -> restrict_data_level) = restrict_data_level;

   /*---------------------------------------------------------------
    * Create the fine patch relax_data structure.
    *---------------------------------------------------------------*/
   relax_data_level   = hypre_TAlloc(void *, max_level+1);
   
   for (level= 0; level<= max_level; level++)
   {
       relax_data_level[level]=  hypre_SysPFMGRelaxCreate(comm);
       hypre_SysPFMGRelaxSetTol(relax_data_level[level], 0.0);
       hypre_SysPFMGRelaxSetType(relax_data_level[level], relax_type);
       if (usr_jacobi_weight)
       {
          hypre_SysPFMGRelaxSetJacobiWeight(relax_data_level[level], jacobi_weight);
       }
       hypre_SysPFMGRelaxSetTempVec(relax_data_level[level], tx_level[level]);
       hypre_SysPFMGRelaxSetup(relax_data_level[level], 
                               hypre_SStructMatrixPMatrix(A_level[level], part_fine),
                               hypre_SStructVectorPVector(b_level[level], part_fine),
                               hypre_SStructVectorPVector(x_level[level], part_fine));
   }
   (fac_data -> relax_data_level)    = relax_data_level;
  
   
   /*---------------------------------------------------------------
    * Create the coarsest composite level preconditioned solver.
    *  csolver_type=   1      multigrid-pcg
    *  csolver_type=   2      multigrid
    *---------------------------------------------------------------*/
   if (csolver_type == 1)
   {
       HYPRE_SStructPCGCreate(comm, &crse_solver);
       HYPRE_PCGSetMaxIter((HYPRE_Solver) crse_solver, 1);
       HYPRE_PCGSetTol((HYPRE_Solver) crse_solver, 1.0e-6);
       HYPRE_PCGSetTwoNorm((HYPRE_Solver) crse_solver, 1);
 
       /* use SysPFMG solver as preconditioner */
       HYPRE_SStructSysPFMGCreate(comm, &crse_precond);
       HYPRE_SStructSysPFMGSetMaxIter(crse_precond, 1);
       HYPRE_SStructSysPFMGSetTol(crse_precond, 0.0);
       HYPRE_SStructSysPFMGSetZeroGuess(crse_precond);
       /* weighted Jacobi = 1; red-black GS = 2 */
       HYPRE_SStructSysPFMGSetRelaxType(crse_precond, 3);
       if (usr_jacobi_weight)
       {
          HYPRE_SStructFACSetJacobiWeight(crse_precond, jacobi_weight);
       }
       HYPRE_SStructSysPFMGSetNumPreRelax(crse_precond, 1);
       HYPRE_SStructSysPFMGSetNumPostRelax(crse_precond, 1);
       HYPRE_PCGSetPrecond((HYPRE_Solver) crse_solver,
                           (HYPRE_PtrToSolverFcn) HYPRE_SStructSysPFMGSolve,
                           (HYPRE_PtrToSolverFcn) HYPRE_SStructSysPFMGSetup,
                           (HYPRE_Solver) crse_precond);

       HYPRE_PCGSetup((HYPRE_Solver) crse_solver, 
                      (HYPRE_Matrix) A_level[0],
                      (HYPRE_Vector) b_level[0],
                      (HYPRE_Vector) x_level[0]);
    }

    else if (csolver_type == 2)
    {
       crse_precond= NULL;

       HYPRE_SStructSysPFMGCreate(comm, &crse_solver);
       HYPRE_SStructSysPFMGSetMaxIter(crse_solver, 1);
       HYPRE_SStructSysPFMGSetTol(crse_solver, 1.0e-6);
       HYPRE_SStructSysPFMGSetZeroGuess(crse_solver);
       /* weighted Jacobi = 1; red-black GS = 2 */
       HYPRE_SStructSysPFMGSetRelaxType(crse_solver, relax_type);
       if (usr_jacobi_weight)
       {
          HYPRE_SStructFACSetJacobiWeight(crse_precond, jacobi_weight);
       }
       HYPRE_SStructSysPFMGSetNumPreRelax(crse_solver, 1);
       HYPRE_SStructSysPFMGSetNumPostRelax(crse_solver, 1);
       HYPRE_SStructSysPFMGSetup(crse_solver, A_level[0], b_level[0], x_level[0]);
    }

   (fac_data -> csolver)  = crse_solver;
   (fac_data -> cprecond) = crse_precond;

    hypre_FacZeroCData(fac_vdata, A_rap);

   return ierr;
}

