/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.12 $
 ***********************************************************************EHEADER*/

/******************************************************************************
 * OpenMP Problems
 *
 * Are private static arrays a problem?
 *
 ******************************************************************************/

#include "headers.h"
#include "maxwell_TV.h"
#include "par_amg.h"

#define DEBUG 0
/*--------------------------------------------------------------------------
 * hypre_MaxwellTV_Setup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MaxwellTV_Setup(void                 *maxwell_vdata,
                      hypre_SStructMatrix  *Aee_in,
                      hypre_SStructVector  *b_in,
                      hypre_SStructVector  *x_in)
{
   hypre_MaxwellData     *maxwell_TV_data = maxwell_vdata;

   MPI_Comm               comm = hypre_SStructMatrixComm(Aee_in);

   hypre_SStructGraph    *graph= hypre_SStructMatrixGraph(Aee_in);
   hypre_SStructGrid     *grid = hypre_SStructGraphGrid(graph);
   hypre_Index           *rfactor_in= (maxwell_TV_data-> rfactor);
   hypre_ParCSRMatrix    *T         = (maxwell_TV_data-> Tgrad);

   hypre_SStructMatrix   *Ann;
   HYPRE_IJMatrix         Aen;
   hypre_SStructVector   *bn;
   hypre_SStructVector   *xn;

   hypre_ParCSRMatrix    *Aee  = hypre_SStructMatrixParCSRMatrix(Aee_in);
   hypre_ParCSRMatrix    *T_transpose;
   hypre_ParCSRMatrix    *transpose;
   hypre_ParCSRMatrix    *parcsr_mat;
   HYPRE_Int              size, *col_inds;
   double                *values;

   hypre_ParVector       *parvector_x;
   hypre_ParVector       *parvector_b;

   hypre_ParCSRMatrix   **Aen_l;

   void                  *amg_vdata;
   hypre_ParAMGData      *amg_data;
   hypre_ParCSRMatrix   **Ann_l;
   hypre_ParCSRMatrix   **Pn_l;
   hypre_ParCSRMatrix   **RnT_l;
   hypre_ParVector      **bn_l;
   hypre_ParVector      **xn_l;
   hypre_ParVector      **resn_l;
   hypre_ParVector      **en_l;
   hypre_ParVector      **nVtemp_l;
   hypre_ParVector      **nVtemp2_l;
   HYPRE_Int            **nCF_marker_l;
   double                *nrelax_weight;
   double                *nomega;
   HYPRE_Int              nrelax_type;
   HYPRE_Int              node_numlevels;

   hypre_ParCSRMatrix   **Aee_l;
   hypre_IJMatrix       **Pe_l; 
   hypre_IJMatrix       **ReT_l;
   hypre_ParVector      **be_l;
   hypre_ParVector      **xe_l;
   hypre_ParVector      **rese_l;
   hypre_ParVector      **ee_l;
   hypre_ParVector      **eVtemp_l;
   hypre_ParVector      **eVtemp2_l;
   double                *erelax_weight;
   double                *eomega;
   HYPRE_Int            **eCF_marker_l;
   HYPRE_Int              erelax_type;

  /* objects needed to fine the edge relaxation parameters */
   HYPRE_Int              relax_type;
   /*HYPRE_Int             *relax_types;
   void                  *e_amg_vdata;
   hypre_ParAMGData      *e_amgData;
   HYPRE_Int              numCGSweeps= 10;
   HYPRE_Int            **amg_CF_marker;
   hypre_ParCSRMatrix   **A_array;*/


   hypre_SStructGrid     *node_grid;
   hypre_SStructGraph    *node_graph;

   HYPRE_Int             *coarsen;
   hypre_SStructGrid    **egrid_l;
   hypre_SStructGrid     *edge_grid, *face_grid, *cell_grid;
   hypre_SStructGrid    **topological_edge, **topological_face, **topological_cell;

   HYPRE_Int            **BdryRanks_l;
   HYPRE_Int             *BdryRanksCnts_l;

   hypre_SStructPGrid    *pgrid;
   hypre_StructGrid      *sgrid;

   hypre_BoxArray        *boxes, *tmp_box_array;
   hypre_Box             *box, *box_piece, *contract_box;
   hypre_BoxArray        *cboxes;

   HYPRE_SStructVariable *vartypes, *vartype_edges, *vartype_faces, *vartype_cell;
   hypre_SStructStencil **Ann_stencils;

   hypre_MaxwellOffProcRow **OffProcRows;
   HYPRE_Int                 num_OffProcRows;

   hypre_Index            rfactor;
   hypre_Index            index, cindex, shape, loop_size, start;
   HYPRE_Int              stencil_size;
   HYPRE_Int              loopi, loopj, loopk;
   HYPRE_Int              matrix_type= HYPRE_PARCSR;

   HYPRE_Int              ndim = hypre_SStructMatrixNDim(Aee_in); 
   HYPRE_Int              nparts, part, vars, nboxes, lev_nboxes;

   HYPRE_Int              nrows, rank, start_rank;
   HYPRE_Int             *flag, *flag2, *inode, *ncols, *jnode;
   double                *vals;

   HYPRE_Int              i, j, k, l, m;

   hypre_BoxManager      *node_boxman;
   hypre_BoxManEntry     *entry;
   HYPRE_Int              kstart, kend;
   HYPRE_Int              ilower, iupper;
   HYPRE_Int              jlower, jupper;
   HYPRE_Int              myproc;

   HYPRE_Int              first_local_row, last_local_row;
   HYPRE_Int              first_local_col, last_local_col;

   HYPRE_Int              edge_maxlevels, edge_numlevels, en_numlevels;

   HYPRE_Int              constant_coef=  maxwell_TV_data -> constant_coef;
   HYPRE_Int              true = 1;
   HYPRE_Int              false= 0;

   HYPRE_Int              ierr = 0;
#if DEBUG
   /*char                  filename[255];*/
#endif

   hypre_MPI_Comm_rank(comm, &myproc);

  (maxwell_TV_data -> ndim)= ndim;

   /* Adjust rfactor so that the correct dimension is used */
   for (i= ndim; i< 3; i++)
   {
      rfactor_in[0][i]= 1;
   }
   hypre_CopyIndex(rfactor_in[0], rfactor);

   /*---------------------------------------------------------------------
    * Set up matrices Ann, Aen.
    *
    * Forming the finest node matrix: We are assuming the Aee_in is in the
    * parcsr data structure, the stencil structure for the node is the
    * 9 or 27 point fem pattern, etc.
    *
    * Need to form the grid, graph, etc. for these matrices.
    *---------------------------------------------------------------------*/
   nparts= hypre_SStructMatrixNParts(Aee_in);
   HYPRE_SStructGridCreate(comm, ndim, nparts, &node_grid);

  /* grids can be constructed from the cell-centre grid of Aee_in */
   vartypes= hypre_CTAlloc(HYPRE_SStructVariable, 1);
   vartypes[0]= HYPRE_SSTRUCT_VARIABLE_NODE;

   for (i= 0; i< nparts; i++)
   {
      pgrid= hypre_SStructPMatrixPGrid(hypre_SStructMatrixPMatrix(Aee_in, i));
      sgrid= hypre_SStructPGridCellSGrid(pgrid);
      
      boxes= hypre_StructGridBoxes(sgrid);
      hypre_ForBoxI(j, boxes)
      {
         box= hypre_BoxArrayBox(boxes, j);
         HYPRE_SStructGridSetExtents(node_grid, i, 
                                     hypre_BoxIMin(box), hypre_BoxIMax(box));
      }
 
      HYPRE_SStructGridSetVariables(node_grid, i, 1, vartypes);
   }
   HYPRE_SStructGridAssemble(node_grid);

  /* Ann stencils & graph */
   stencil_size= 1;
   for (i= 0; i< ndim; i++)
   {
      stencil_size*= 3;
   }

   Ann_stencils= hypre_CTAlloc(hypre_SStructStencil *, 1);
   HYPRE_SStructStencilCreate(ndim, stencil_size, &Ann_stencils[0]);

   vars= 0; /* scalar equation, node-to-node */
   if (ndim > 2)
   {
      kstart= -1; 
      kend  =  2;
   }
   else if (ndim == 2)
   {
      kstart= 0; 
      kend  = 1;
   }
      
   m= 0;
   for (k= kstart; k< kend; k++)
   {
      for (j= -1; j< 2; j++)
      {
         for (i= -1; i< 2; i++)
         {
            hypre_SetIndex(shape, i, j, k);
            HYPRE_SStructStencilSetEntry(Ann_stencils[0], m, shape, vars);
            m++;
         }
      }
   }

   HYPRE_SStructGraphCreate(comm, node_grid, &node_graph);
   for (part= 0; part< nparts; part++)
   {
      HYPRE_SStructGraphSetStencil(node_graph, part, 0, Ann_stencils[0]);
   }
   HYPRE_SStructGraphAssemble(node_graph);

   HYPRE_SStructMatrixCreate(comm, node_graph, &Ann);
   HYPRE_SStructMatrixSetObjectType(Ann, HYPRE_PARCSR);
   HYPRE_SStructMatrixInitialize(Ann);

  /* Aen is constructed as an IJ matrix. Constructing it as a sstruct_matrix
   * would make it a square matrix. */
   part= 0;
   i   = 0;

   hypre_SStructGridBoxProcFindBoxManEntry(node_grid, part, 0, i, myproc, &entry);
   pgrid= hypre_SStructGridPGrid(node_grid, part);
   vartypes[0]= HYPRE_SSTRUCT_VARIABLE_NODE;
   j= vartypes[0];
   sgrid= hypre_SStructPGridVTSGrid(pgrid, j);
   boxes= hypre_StructGridBoxes(sgrid);
   box  = hypre_BoxArrayBox(boxes, 0);
   hypre_SStructBoxManEntryGetGlobalCSRank(entry, hypre_BoxIMin(box), &jlower);

   hypre_SStructGridBoxProcFindBoxManEntry(grid, part, 0, i, myproc, &entry);
   pgrid= hypre_SStructGridPGrid(grid, part);
  /* grab the first edge variable type */
   vartypes[0]= hypre_SStructPGridVarType(pgrid, 0);
   j= vartypes[0];
   sgrid= hypre_SStructPGridVTSGrid(pgrid, j);
   boxes= hypre_StructGridBoxes(sgrid);
   box  = hypre_BoxArrayBox(boxes, 0);
   hypre_SStructBoxManEntryGetGlobalCSRank(entry, hypre_BoxIMin(box), &ilower);

   part = nparts-1;
   pgrid= hypre_SStructGridPGrid(node_grid, part);
   vartypes[0]= HYPRE_SSTRUCT_VARIABLE_NODE;
   j= vartypes[0];
   sgrid= hypre_SStructPGridVTSGrid(pgrid, j);
   boxes= hypre_StructGridBoxes(sgrid);
   box  = hypre_BoxArrayBox(boxes, hypre_BoxArraySize(boxes)-1);

   hypre_SStructGridBoxProcFindBoxManEntry(node_grid, part, 0, 
                                        hypre_BoxArraySize(boxes)-1,
                                        myproc, &entry);
   hypre_SStructBoxManEntryGetGlobalCSRank(entry, hypre_BoxIMax(box), &jupper);

   pgrid= hypre_SStructGridPGrid(grid, part);
   vars = hypre_SStructPGridNVars(pgrid);
   vartypes[0]= hypre_SStructPGridVarType(pgrid, vars-1);
   j= vartypes[0];
   sgrid= hypre_SStructPGridVTSGrid(pgrid, j);
   boxes= hypre_StructGridBoxes(sgrid);
   box  = hypre_BoxArrayBox(boxes, hypre_BoxArraySize(boxes)-1);
   hypre_TFree(vartypes);

   hypre_SStructGridBoxProcFindBoxManEntry(grid, part, vars-1, 
                                        hypre_BoxArraySize(boxes)-1,
                                        myproc, &entry);
   hypre_SStructBoxManEntryGetGlobalCSRank(entry, hypre_BoxIMax(box), &iupper);

   HYPRE_IJMatrixCreate(comm, ilower, iupper, jlower, jupper, &Aen);
   HYPRE_IJMatrixSetObjectType(Aen, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(Aen);

   /* setup the Aen & Ann using matrix-matrix products 
    * Aen's parscr matrix has not been formed yet-> fill up ij_matrix */
   parcsr_mat= hypre_ParMatmul(Aee, T);
   HYPRE_ParCSRMatrixGetLocalRange((HYPRE_ParCSRMatrix) parcsr_mat, 
                                  &first_local_row, &last_local_row,
                                  &first_local_col, &last_local_col);

   for (i= first_local_row; i<= last_local_row; i++)
   {
      HYPRE_ParCSRMatrixGetRow((HYPRE_ParCSRMatrix) parcsr_mat, 
                               i, &size, &col_inds, &values);
      HYPRE_IJMatrixSetValues(Aen, 1, &size, &i, (const HYPRE_Int *) col_inds,
                             (const double *) values);
      HYPRE_ParCSRMatrixRestoreRow((HYPRE_ParCSRMatrix) parcsr_mat, 
                                    i, &size, &col_inds, &values);
   }
   hypre_ParCSRMatrixDestroy(parcsr_mat);
   HYPRE_IJMatrixAssemble(Aen);

   /* Ann's parscr matrix has not been formed yet-> fill up ij_matrix */
   hypre_ParCSRMatrixTranspose(T, &T_transpose, 1);
   parcsr_mat= hypre_ParMatmul(T_transpose, 
                     (hypre_ParCSRMatrix *) hypre_IJMatrixObject(Aen));
   HYPRE_ParCSRMatrixGetLocalRange((HYPRE_ParCSRMatrix) parcsr_mat, 
                                  &first_local_row, &last_local_row,
                                  &first_local_col, &last_local_col);

   for (i= first_local_row; i<= last_local_row; i++)
   {
      HYPRE_ParCSRMatrixGetRow((HYPRE_ParCSRMatrix) parcsr_mat, 
                               i, &size, &col_inds, &values);
      HYPRE_IJMatrixSetValues(hypre_SStructMatrixIJMatrix(Ann),
                              1, &size, &i, (const HYPRE_Int *) col_inds,
                             (const double *) values);
      HYPRE_ParCSRMatrixRestoreRow((HYPRE_ParCSRMatrix) parcsr_mat, 
                                    i, &size, &col_inds, &values);
   }
   hypre_ParCSRMatrixDestroy(parcsr_mat);

   /* set the physical boundary points to identity */
   nrows= 0;
   for (part= 0; part< nparts; part++)
   {
      pgrid = hypre_SStructGridPGrid(node_grid, part);
      sgrid = hypre_SStructPGridSGrid(pgrid, 0);
      nrows+= hypre_StructGridLocalSize(sgrid);
   }

   flag = hypre_CTAlloc(HYPRE_Int, nrows);
   flag2= hypre_CTAlloc(HYPRE_Int, nrows);
   for (i= 0; i< nrows; i++)
   {
      flag[i]= 1;
   }

   /* Determine physical boundary points. Get the rank and set flag[rank]= rank.
      This will boundary point, i.e., ncols[rank]> 0 will flag a boundary point. */
   start_rank= hypre_SStructGridStartRank(node_grid);
   for (part= 0; part< nparts; part++)
   {
      pgrid   = hypre_SStructGridPGrid(node_grid, part);
      sgrid   = hypre_SStructPGridSGrid(pgrid, 0);
      boxes   = hypre_StructGridBoxes(sgrid);
      node_boxman = hypre_SStructGridBoxManager(node_grid, part, 0);

      hypre_ForBoxI(j, boxes)
      {
         box= hypre_BoxArrayBox(boxes, j);
         hypre_BoxManGetEntry(node_boxman, myproc, j, &entry);
         i= hypre_BoxVolume(box);

         tmp_box_array= hypre_BoxArrayCreate(0);
         ierr        += hypre_BoxBoundaryG(box, sgrid, tmp_box_array);

         for (m= 0; m< hypre_BoxArraySize(tmp_box_array); m++)
         {
            box_piece= hypre_BoxArrayBox(tmp_box_array, m);
            if (hypre_BoxVolume(box_piece) < i)
            {
               hypre_BoxGetSize(box_piece, loop_size);
               hypre_CopyIndex(hypre_BoxIMin(box_piece), start);

               hypre_BoxLoop0Begin(loop_size);
#if 0 /* Are private static arrays a problem? */
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,index,rank
#include "hypre_box_smp_forloop.h"
#endif
               hypre_BoxLoop0For(loopi, loopj, loopk)
               {
                   hypre_SetIndex(index, loopi, loopj, loopk);
                   hypre_AddIndex(index, start, index);

                   hypre_SStructBoxManEntryGetGlobalRank(entry, index,
                                                     &rank, matrix_type);
                   flag[rank-start_rank] = 0;
                   flag2[rank-start_rank]= rank;
               }
               hypre_BoxLoop0End();
            }  /* if (hypre_BoxVolume(box_piece) < i) */
         }  /* for (m= 0; m< hypre_BoxArraySize(tmp_box_array); m++) */
         hypre_BoxArrayDestroy(tmp_box_array);
      }  /* hypre_ForBoxI(j, boxes) */
   }     /* for (part= 0; part< nparts; part++) */

   /* set up boundary identity */
   j= 0;
   for (i= 0; i< nrows; i++)
   {
     if (!flag[i])
     {
        j++;
     }
   }

   inode= hypre_CTAlloc(HYPRE_Int, j);
   ncols= hypre_CTAlloc(HYPRE_Int, j);
   jnode= hypre_CTAlloc(HYPRE_Int, j);
   vals = hypre_TAlloc(double, j);

   j= 0;
   for (i= 0; i< nrows; i++)
   {
     if (!flag[i])
     {
        ncols[j]= 1;
        inode[j]= flag2[i];
        jnode[j]= flag2[i];
        vals[j] = 1.0;
        j++;
     }
   }
   hypre_TFree(flag);
   hypre_TFree(flag2);

   HYPRE_IJMatrixSetValues(hypre_SStructMatrixIJMatrix(Ann),
                           j, ncols, (const HYPRE_Int*) inode,
                          (const HYPRE_Int*) jnode, (const double*) vals);
   hypre_TFree(ncols);
   hypre_TFree(inode);
   hypre_TFree(jnode);
   hypre_TFree(vals);

   HYPRE_SStructMatrixAssemble(Ann);
#if DEBUG
      HYPRE_SStructMatrixPrint("sstruct.out.Ann",  Ann, 0);
      HYPRE_IJMatrixPrint(Aen, "driver.out.Aen");
#endif

   /* setup bn & xn using matvec. Assemble first and then perform matvec to get
      the nodal rhs and initial guess. */
   HYPRE_SStructVectorCreate(comm, node_grid, &bn);
   HYPRE_SStructVectorSetObjectType(bn, HYPRE_PARCSR);
   HYPRE_SStructVectorInitialize(bn);
   HYPRE_SStructVectorAssemble(bn);

   hypre_SStructVectorConvert(b_in, &parvector_x);
   /*HYPRE_SStructVectorGetObject((HYPRE_SStructVector) b_in, (void **) &parvector_x);*/
   HYPRE_SStructVectorGetObject((HYPRE_SStructVector) bn, (void **) &parvector_b);
   hypre_ParCSRMatrixMatvec(1.0, T_transpose, parvector_x, 0.0, parvector_b);
   
   HYPRE_SStructVectorCreate(comm, node_grid, &xn);
   HYPRE_SStructVectorSetObjectType(xn, HYPRE_PARCSR);
   HYPRE_SStructVectorInitialize(xn);
   HYPRE_SStructVectorAssemble(xn);
   
   hypre_SStructVectorConvert(x_in, &parvector_x);
   /*HYPRE_SStructVectorGetObject((HYPRE_SStructVector) x_in, (void **) &parvector_x);*/
   HYPRE_SStructVectorGetObject((HYPRE_SStructVector) xn, (void **) &parvector_b);
   hypre_ParCSRMatrixMatvec(1.0, T_transpose, parvector_x, 0.0, parvector_b);

   /* Destroy the node grid and graph. This only decrements reference counters. */
   HYPRE_SStructGridDestroy(node_grid);
   HYPRE_SStructGraphDestroy(node_graph);

   /* create the multigrid components for the nodal matrix using amg. We need
      to extract the nodal mg components to form the system mg components. */
   amg_vdata= (void *) hypre_BoomerAMGCreate();
   hypre_BoomerAMGSetStrongThreshold(amg_vdata, 0.25);
   hypre_BoomerAMGSetup(amg_vdata, 
                        hypre_SStructMatrixParCSRMatrix(Ann),
                        hypre_SStructVectorParVector(bn),
                        hypre_SStructVectorParVector(xn));
   {
       amg_data = amg_vdata;

       node_numlevels= hypre_ParAMGDataNumLevels(amg_data);

       Ann_l   = hypre_CTAlloc(hypre_ParCSRMatrix *, node_numlevels);
       Pn_l    = hypre_CTAlloc(hypre_ParCSRMatrix *, node_numlevels);
       RnT_l   = hypre_CTAlloc(hypre_ParCSRMatrix *, node_numlevels);
       bn_l    = hypre_CTAlloc(hypre_ParVector*, node_numlevels);
       xn_l    = hypre_CTAlloc(hypre_ParVector*, node_numlevels);
       resn_l  = hypre_CTAlloc(hypre_ParVector*, node_numlevels);
       en_l    = hypre_CTAlloc(hypre_ParVector*, node_numlevels);
       nVtemp_l= hypre_CTAlloc(hypre_ParVector*, node_numlevels);
       nVtemp2_l= hypre_CTAlloc(hypre_ParVector*, node_numlevels);

      /* relaxation parameters */
       nCF_marker_l = hypre_CTAlloc(HYPRE_Int *, node_numlevels);
       nrelax_weight= hypre_CTAlloc(double , node_numlevels);
       nomega       = hypre_CTAlloc(double , node_numlevels);
       nrelax_type  = 6;  /* fast parallel hybrid */

       for (i= 0; i< node_numlevels; i++)
       {
          Ann_l[i]= (hypre_ParAMGDataAArray(amg_data))[i];
          Pn_l[i] = hypre_ParAMGDataPArray(amg_data)[i];
          RnT_l[i]= hypre_ParAMGDataRArray(amg_data)[i];
       
          bn_l[i] = hypre_ParAMGDataFArray(amg_data)[i];
          xn_l[i] = hypre_ParAMGDataUArray(amg_data)[i];

        /* create temporary vectors */
          resn_l[i]= hypre_ParVectorCreate(hypre_ParCSRMatrixComm(Ann_l[i]),
                                 hypre_ParCSRMatrixGlobalNumRows(Ann_l[i]),
                                 hypre_ParCSRMatrixRowStarts(Ann_l[i]));
          hypre_ParVectorInitialize(resn_l[i]);
          hypre_ParVectorSetPartitioningOwner(resn_l[i], 0);

          en_l[i]= hypre_ParVectorCreate(hypre_ParCSRMatrixComm(Ann_l[i]),
                                 hypre_ParCSRMatrixGlobalNumRows(Ann_l[i]),
                                 hypre_ParCSRMatrixRowStarts(Ann_l[i]));
          hypre_ParVectorInitialize(en_l[i]);
          hypre_ParVectorSetPartitioningOwner(en_l[i], 0);

          nVtemp_l[i]= hypre_ParVectorCreate(hypre_ParCSRMatrixComm(Ann_l[i]),
                                 hypre_ParCSRMatrixGlobalNumRows(Ann_l[i]),
                                 hypre_ParCSRMatrixRowStarts(Ann_l[i]));
          hypre_ParVectorInitialize(nVtemp_l[i]);
          hypre_ParVectorSetPartitioningOwner(nVtemp_l[i], 0);

          nVtemp2_l[i]= hypre_ParVectorCreate(hypre_ParCSRMatrixComm(Ann_l[i]),
                                 hypre_ParCSRMatrixGlobalNumRows(Ann_l[i]),
                                 hypre_ParCSRMatrixRowStarts(Ann_l[i]));
          hypre_ParVectorInitialize(nVtemp2_l[i]);
          hypre_ParVectorSetPartitioningOwner(nVtemp2_l[i], 0);

          nCF_marker_l[i] = hypre_ParAMGDataCFMarkerArray(amg_data)[i];
          nrelax_weight[i]= hypre_ParAMGDataRelaxWeight(amg_data)[i];
          nomega[i]       = hypre_ParAMGDataOmega(amg_data)[i];
       }
   }
  (maxwell_TV_data -> Ann_stencils)    = Ann_stencils;
  (maxwell_TV_data -> T_transpose)     = T_transpose;
  (maxwell_TV_data -> Ann)             = Ann;
  (maxwell_TV_data -> Aen)             = Aen;
  (maxwell_TV_data -> bn)              = bn;
  (maxwell_TV_data -> xn)              = xn;

  (maxwell_TV_data -> amg_vdata)       = amg_vdata;
  (maxwell_TV_data -> Ann_l)           = Ann_l;
  (maxwell_TV_data -> Pn_l)            = Pn_l;
  (maxwell_TV_data -> RnT_l)           = RnT_l;
  (maxwell_TV_data -> bn_l)            = bn_l;
  (maxwell_TV_data -> xn_l)            = xn_l;
  (maxwell_TV_data -> resn_l)          = resn_l;
  (maxwell_TV_data -> en_l)            = en_l;
  (maxwell_TV_data -> nVtemp_l)        = nVtemp_l;
  (maxwell_TV_data -> nVtemp2_l)       = nVtemp2_l;
  (maxwell_TV_data -> nCF_marker_l)    = nCF_marker_l;
  (maxwell_TV_data -> nrelax_weight)   = nrelax_weight;
  (maxwell_TV_data -> nomega)          = nomega;
  (maxwell_TV_data -> nrelax_type)     = nrelax_type;
  (maxwell_TV_data -> node_numlevels)  = node_numlevels;

   /* coarsen the edge matrix. Will coarsen uniformly since we have no 
    * scheme to semi-coarsen. That is, coarsen wrt to rfactor, with
    * rfactor[i] > 1 for i < ndim.
    * Determine the number of levels for the edge problem */
   cboxes= hypre_BoxArrayCreate(0);
   coarsen= hypre_CTAlloc(HYPRE_Int, nparts);
   edge_maxlevels= 0;
   for (part= 0; part< nparts; part++)
   {
      pgrid= hypre_SStructGridPGrid(grid, part);
      sgrid= hypre_SStructPGridCellSGrid(pgrid);

      box= hypre_BoxDuplicate(hypre_StructGridBoundingBox(sgrid));
      hypre_AppendBox(box, cboxes);
     /* since rfactor[i]>1, the following i will be an upper bound of
        the number of levels. */
      i  = hypre_Log2(hypre_BoxSizeD(box, 0)) + 2 +
           hypre_Log2(hypre_BoxSizeD(box, 1)) + 2 +
           hypre_Log2(hypre_BoxSizeD(box, 2)) + 2;

      hypre_BoxDestroy(box);
     /* the following allows some of the parts to have volume zero grids */
      edge_maxlevels= hypre_max(edge_maxlevels, i);
      coarsen[part] = true;
   }

   if ((maxwell_TV_data-> edge_maxlevels) > 0)
   {
      edge_maxlevels= hypre_min(edge_maxlevels, 
                            (maxwell_TV_data -> edge_maxlevels));
   }

  (maxwell_TV_data -> edge_maxlevels)= edge_maxlevels;

  /* form the edge grids: coarsen the cell grid on each part and then
     set the boxes of these grids to be the boxes of the sstruct_grid. */
   egrid_l   = hypre_TAlloc(hypre_SStructGrid *, edge_maxlevels);
   hypre_SStructGridRef(grid, &egrid_l[0]);

  /* form the topological grids for the topological matrices. */

  /* Assuming same variable ordering on all parts */
   pgrid= hypre_SStructGridPGrid(grid, 0);

   HYPRE_SStructGridCreate(comm, ndim, nparts, &edge_grid);
   vartype_edges= hypre_CTAlloc(HYPRE_SStructVariable, ndim);
   if (ndim > 2)
   {
       HYPRE_SStructGridCreate(comm, ndim, nparts, &face_grid);
       vartype_faces= hypre_CTAlloc(HYPRE_SStructVariable, ndim);
       for (i= 0; i< 3; i++)
       {
          vartype_edges[2]= hypre_SStructPGridVarType(pgrid, i);
          j= vartype_edges[2];

          switch(j)
          {
             case 5:
             {
                vartype_edges[i]= HYPRE_SSTRUCT_VARIABLE_XEDGE;
                vartype_faces[i]= HYPRE_SSTRUCT_VARIABLE_XFACE;
                break;
             }
             case 6:
             {
                vartype_edges[i]= HYPRE_SSTRUCT_VARIABLE_YEDGE;
                vartype_faces[i]= HYPRE_SSTRUCT_VARIABLE_YFACE;
                break;
             }
             case 7:
             {
                vartype_edges[i]= HYPRE_SSTRUCT_VARIABLE_ZEDGE;
                vartype_faces[i]= HYPRE_SSTRUCT_VARIABLE_ZFACE;
                break;
             }

          }  /* switch(j) */
       }     /* for (i= 0; i< 3; i++) */
   }
   else
   {
      for (i= 0; i< 2; i++)
      {
         vartype_edges[1]= hypre_SStructPGridVarType(pgrid, i);
         j= vartype_edges[1];

         switch(j)
         {
            case 2:
            {
               vartype_edges[i]= HYPRE_SSTRUCT_VARIABLE_XFACE;
               break;
            }
            case 3:
            {
               vartype_edges[i]= HYPRE_SSTRUCT_VARIABLE_YFACE;
               break;
            }
         }  /* switch(j) */
      }     /* for (i= 0; i< 3; i++) */
   }

   HYPRE_SStructGridCreate(comm, ndim, nparts, &cell_grid);
   vartype_cell= hypre_CTAlloc(HYPRE_SStructVariable, 1);
   vartype_cell[0]= HYPRE_SSTRUCT_VARIABLE_CELL;

   for (i= 0; i< nparts; i++)
   {
      pgrid= hypre_SStructPMatrixPGrid(hypre_SStructMatrixPMatrix(Aee_in, i));
      sgrid= hypre_SStructPGridCellSGrid(pgrid);
      
      boxes= hypre_StructGridBoxes(sgrid);
      hypre_ForBoxI(j, boxes)
      {
          box= hypre_BoxArrayBox(boxes, j);
          HYPRE_SStructGridSetExtents(edge_grid, i, 
                                      hypre_BoxIMin(box), hypre_BoxIMax(box));
          HYPRE_SStructGridSetExtents(cell_grid, i, 
                                      hypre_BoxIMin(box), hypre_BoxIMax(box));
          if (ndim > 2)
          {
             HYPRE_SStructGridSetExtents(face_grid, i, 
                                         hypre_BoxIMin(box), hypre_BoxIMax(box));
          }
      }
      HYPRE_SStructGridSetVariables(edge_grid, i, ndim, vartype_edges);
      HYPRE_SStructGridSetVariables(cell_grid, i, 1, vartype_cell);

      if (ndim > 2)
      {
         HYPRE_SStructGridSetVariables(face_grid, i, ndim, vartype_faces);
      }
   }

   HYPRE_SStructGridAssemble(edge_grid);
   topological_edge   = hypre_TAlloc(hypre_SStructGrid *, edge_maxlevels);
   topological_edge[0]= edge_grid;

   HYPRE_SStructGridAssemble(cell_grid);
   topological_cell   = hypre_TAlloc(hypre_SStructGrid *, edge_maxlevels);
   topological_cell[0]= cell_grid;

   if (ndim > 2)
   {
       HYPRE_SStructGridAssemble(face_grid);
       topological_face= hypre_TAlloc(hypre_SStructGrid *, edge_maxlevels);
       topological_face[0]= face_grid;
   }

  /*--------------------------------------------------------------------------
   * to determine when to stop coarsening, we check the cell bounding boxes
   * of the level egrid. After each coarsening, the bounding boxes are 
   * replaced by the generated coarse egrid cell bounding boxes.
   *--------------------------------------------------------------------------*/
   hypre_SetIndex(cindex, 0, 0, 0);
   j= 0; /* j tracks the number of parts that have been coarsened away */
   edge_numlevels= 1;

   for (l= 0; ; l++)
   {
      HYPRE_SStructGridCreate(comm, ndim, nparts, &egrid_l[l+1]);
      HYPRE_SStructGridCreate(comm, ndim, nparts, &topological_edge[l+1]);
      HYPRE_SStructGridCreate(comm, ndim, nparts, &topological_cell[l+1]);
      if (ndim > 2)
      {
          HYPRE_SStructGridCreate(comm, ndim, nparts, &topological_face[l+1]);
      }

      /* coarsen the non-zero bounding boxes only if we have some. */
      nboxes= 0;
      if (j < nparts)
      {
         for (part= 0; part< nparts; part++)
         {
            pgrid= hypre_SStructGridPGrid(egrid_l[l], part);
            sgrid= hypre_SStructPGridCellSGrid(pgrid);

            if (coarsen[part])
            {
               box= hypre_BoxArrayBox(cboxes, part);
               m= true;
               for (i= 0; i< ndim; i++)
               {
                  if ( hypre_BoxIMaxD(box, i) < hypre_BoxIMinD(box, i) )
                  {
                     m= false;
                     break;
                  }
               }
              
               if (m)
               {
/*   MAY NEED TO CHECK THE FOLLOWING MORE CAREFULLY: */
                 /* should we decrease this bounding box so that we get the
                    correct coarse bounding box? Recall that we will decrease
                    each box of the cell_grid so that exact rfactor divisibility
                    is attained. Project does not automatically perform this.
                    E.g., consider a grid with only one box whose width
                    does not divide by rfactor, but it contains beginning and
                    ending indices that are divisible by rfactor. Then an extra
                    coarse grid layer is given by project. */

                  contract_box= hypre_BoxContraction(box, sgrid, rfactor);
                  hypre_CopyBox(contract_box, box);
                  hypre_BoxDestroy(contract_box);

                  hypre_ProjectBox(box, cindex, rfactor);
                  hypre_StructMapFineToCoarse(hypre_BoxIMin(box), cindex, 
                                              rfactor, hypre_BoxIMin(box));
                  hypre_StructMapFineToCoarse(hypre_BoxIMax(box), cindex, 
                                              rfactor, hypre_BoxIMax(box));

                 /* build the coarse edge grids. Only fill up box extents. 
                    The boxes of the grid may be contracted. Note that the
                    box projection may not perform the contraction. */
                  k= 0;
                  hypre_CoarsenPGrid(egrid_l[l], cindex, rfactor, part,
                                     egrid_l[l+1], &k);

                 /* build the topological grids */
                  hypre_CoarsenPGrid(topological_edge[l], cindex, rfactor, part, 
                                     topological_edge[l+1], &i);
                  hypre_CoarsenPGrid(topological_cell[l], cindex, rfactor, part, 
                                     topological_cell[l+1], &i);
                  if (ndim > 2)
                  {
                      hypre_CoarsenPGrid(topological_face[l], cindex, rfactor, 
                                         part, topological_face[l+1], &i);
                  }
                  nboxes+= k;
               }
               else 
               {
                 /* record empty, coarsened-away part */
                  coarsen[part]= false;
                 /* set up a dummy box so this grid can be destroyed */
                  HYPRE_SStructGridSetExtents(egrid_l[l+1], part,
                                  hypre_BoxIMin(box), hypre_BoxIMin(box));
                     
                  HYPRE_SStructGridSetExtents(topological_edge[l+1], part,
                                  hypre_BoxIMin(box), hypre_BoxIMin(box));
                     
                  HYPRE_SStructGridSetExtents(topological_cell[l+1], part,
                                  hypre_BoxIMin(box), hypre_BoxIMin(box));

                  if (ndim > 2)
                  {
                     HYPRE_SStructGridSetExtents(topological_face[l+1], part,
                                  hypre_BoxIMin(box), hypre_BoxIMin(box));
                  }   
                  j++;
               }

            }  /* if (coarsen[part]) */

            vartypes= hypre_SStructPGridVarTypes(
                            hypre_SStructGridPGrid(egrid_l[l], part));
            HYPRE_SStructGridSetVariables(egrid_l[l+1], part, ndim, 
                                          vartypes);
                                          
            HYPRE_SStructGridSetVariables(topological_edge[l+1], part, ndim, 
                                          vartype_edges);
            HYPRE_SStructGridSetVariables(topological_cell[l+1], part, 1, 
                                          vartype_cell);
            if (ndim > 2)
            {
                HYPRE_SStructGridSetVariables(topological_face[l+1], part, ndim, 
                                              vartype_faces);
            }
         }  /* for (part= 0; part< nparts; part++) */
      }     /* if (j < nparts) */
   
      HYPRE_SStructGridAssemble(egrid_l[l+1]);
      HYPRE_SStructGridAssemble(topological_edge[l+1]);
      HYPRE_SStructGridAssemble(topological_cell[l+1]);
      if (ndim > 2)
      {
         HYPRE_SStructGridAssemble(topological_face[l+1]);
      }

      lev_nboxes= 0;
      hypre_MPI_Allreduce(&nboxes, &lev_nboxes, 1, HYPRE_MPI_INT, hypre_MPI_SUM,
                    hypre_SStructGridComm(egrid_l[l+1]));

      if (lev_nboxes)  /* there were coarsen boxes */
      {
         edge_numlevels++;
      }

      else
      {
        /* no coarse boxes. Trigger coarsening completed and destroy the
           cgrids corresponding to this level. */
         j= nparts;
      }

     /* extract the cell bounding boxes */
      if (j < nparts)
      {
         for (part= 0; part< nparts; part++)
         {
            if (coarsen[part])
            {
               pgrid= hypre_SStructGridPGrid(egrid_l[l+1], part);
               sgrid= hypre_SStructPGridCellSGrid(pgrid);

               box= hypre_BoxDuplicate(hypre_StructGridBoundingBox(sgrid));
               hypre_CopyBox(box, hypre_BoxArrayBox(cboxes,part));
               hypre_BoxDestroy(box);
            }
         }
      }

      else
      {
         HYPRE_SStructGridDestroy(egrid_l[l+1]);
         HYPRE_SStructGridDestroy(topological_edge[l+1]);
         HYPRE_SStructGridDestroy(topological_cell[l+1]);
         if (ndim > 2)
         {
            HYPRE_SStructGridDestroy(topological_face[l+1]);
         }
         break;
      }
   }
  (maxwell_TV_data -> egrid_l)= egrid_l;

   hypre_Maxwell_PhysBdy(egrid_l, edge_numlevels, rfactor,
                        &BdryRanks_l, &BdryRanksCnts_l);

  (maxwell_TV_data -> BdryRanks_l)    = BdryRanks_l;
  (maxwell_TV_data -> BdryRanksCnts_l)= BdryRanksCnts_l;

   hypre_BoxArrayDestroy(cboxes);
   hypre_TFree(coarsen);
   /* okay to de-allocate vartypes now */
   hypre_TFree(vartype_edges);
   hypre_TFree(vartype_cell);
   if (ndim > 2)
   {
       hypre_TFree(vartype_faces);
   }


  /* Aen matrices are defined for min(edge_numlevels, node_numlevels). */
   en_numlevels= hypre_min(edge_numlevels, node_numlevels);
  (maxwell_TV_data -> en_numlevels)  = en_numlevels;
  (maxwell_TV_data -> edge_numlevels)= edge_numlevels;

   Aee_l= hypre_TAlloc(hypre_ParCSRMatrix *, edge_numlevels);
   Aen_l= hypre_TAlloc(hypre_ParCSRMatrix *, en_numlevels);

  /* Pe_l are defined to be IJ matrices rather than directly parcsr. This
     was done so that in the topological formation, some of the ij matrix
     routines can be used. */
   Pe_l    = hypre_TAlloc(hypre_IJMatrix  *, edge_numlevels-1);
   ReT_l   = hypre_TAlloc(hypre_IJMatrix  *, edge_numlevels-1);

   be_l    = hypre_TAlloc(hypre_ParVector *, edge_numlevels);
   xe_l    = hypre_TAlloc(hypre_ParVector *, edge_numlevels);
   rese_l  = hypre_TAlloc(hypre_ParVector *, edge_numlevels);
   ee_l    = hypre_TAlloc(hypre_ParVector *, edge_numlevels);
   eVtemp_l= hypre_TAlloc(hypre_ParVector *, edge_numlevels);
   eVtemp2_l= hypre_TAlloc(hypre_ParVector *, edge_numlevels);

   Aee_l[0]= hypre_SStructMatrixParCSRMatrix(Aee_in);
   Aen_l[0]=(hypre_ParCSRMatrix *) hypre_IJMatrixObject(Aen), 
   be_l[0] = hypre_SStructVectorParVector(b_in);
   xe_l[0] = hypre_SStructVectorParVector(x_in);

   rese_l[0]=
             hypre_ParVectorCreate(hypre_ParCSRMatrixComm(Aee_l[0]),
                                   hypre_ParCSRMatrixGlobalNumRows(Aee_l[0]),
                                   hypre_ParCSRMatrixRowStarts(Aee_l[0]));
   hypre_ParVectorInitialize(rese_l[0]);
   hypre_ParVectorSetPartitioningOwner(rese_l[0], 0);

   ee_l[0]=
           hypre_ParVectorCreate(hypre_ParCSRMatrixComm(Aee_l[0]),
                                 hypre_ParCSRMatrixGlobalNumRows(Aee_l[0]),
                                 hypre_ParCSRMatrixRowStarts(Aee_l[0]));
   hypre_ParVectorInitialize(ee_l[0]);
   hypre_ParVectorSetPartitioningOwner(ee_l[0], 0);

   eVtemp_l[0]=
             hypre_ParVectorCreate(hypre_ParCSRMatrixComm(Aee_l[0]),
                                   hypre_ParCSRMatrixGlobalNumRows(Aee_l[0]),
                                   hypre_ParCSRMatrixRowStarts(Aee_l[0]));
   hypre_ParVectorInitialize(eVtemp_l[0]);
   hypre_ParVectorSetPartitioningOwner(eVtemp_l[0], 0);

   eVtemp2_l[0]=
             hypre_ParVectorCreate(hypre_ParCSRMatrixComm(Aee_l[0]),
                                   hypre_ParCSRMatrixGlobalNumRows(Aee_l[0]),
                                   hypre_ParCSRMatrixRowStarts(Aee_l[0]));
   hypre_ParVectorInitialize(eVtemp2_l[0]);
   hypre_ParVectorSetPartitioningOwner(eVtemp2_l[0], 0);

   for (l = 0; l < (en_numlevels - 1); l++)
   {
      if (l < edge_numlevels) /* create edge operators */
      {
         if (!constant_coef)
         {
            void             *PTopology_vdata;
            hypre_PTopology  *PTopology;

            hypre_CreatePTopology(&PTopology_vdata);
            if (ndim > 2)
            {
               Pe_l[l]= hypre_Maxwell_PTopology(topological_edge[l],
                                                topological_edge[l+1],
                                                topological_face[l],
                                                topological_face[l+1],
                                                topological_cell[l],
                                                topological_cell[l+1],
                                                Aee_l[l],
                                                rfactor,
                                                PTopology_vdata);
            }
            else
            {
            /* two-dim case: edges= faces but stored in edge grid */
               Pe_l[l]= hypre_Maxwell_PTopology(topological_edge[l],
                                                topological_edge[l+1],
                                                topological_edge[l],
                                                topological_edge[l+1],
                                                topological_cell[l],
                                                topological_cell[l+1],
                                                Aee_l[l],
                                                rfactor,
                                                PTopology_vdata);
            }

            PTopology= PTopology_vdata;

           /* extract off-processors rows of Pe_l[l]. Needed for amge.*/
            hypre_SStructSharedDOF_ParcsrMatRowsComm(egrid_l[l],
                 (hypre_ParCSRMatrix *) hypre_IJMatrixObject(Pe_l[l]),
                                       &num_OffProcRows,
                                       &OffProcRows);

            if (ndim == 3)
            {
               hypre_ND1AMGeInterpolation(Aee_l[l],
                 (hypre_ParCSRMatrix *) hypre_IJMatrixObject(PTopology -> Element_iedge),
                 (hypre_ParCSRMatrix *) hypre_IJMatrixObject(PTopology -> Face_iedge),
                 (hypre_ParCSRMatrix *) hypre_IJMatrixObject(PTopology -> Edge_iedge),
                 (hypre_ParCSRMatrix *) hypre_IJMatrixObject(PTopology -> Element_Face),
                 (hypre_ParCSRMatrix *) hypre_IJMatrixObject(PTopology -> Element_Edge),
                  num_OffProcRows,
                  OffProcRows,
                  Pe_l[l]);
            }
            else
            {
               hypre_ND1AMGeInterpolation(Aee_l[l],
                 (hypre_ParCSRMatrix *) hypre_IJMatrixObject(PTopology -> Element_iedge),
                 (hypre_ParCSRMatrix *) hypre_IJMatrixObject(PTopology -> Edge_iedge),
                 (hypre_ParCSRMatrix *) hypre_IJMatrixObject(PTopology -> Edge_iedge),
                 (hypre_ParCSRMatrix *) hypre_IJMatrixObject(PTopology -> Element_Edge),
                 (hypre_ParCSRMatrix *) hypre_IJMatrixObject(PTopology -> Element_Edge),
                  num_OffProcRows,
                  OffProcRows,
                  Pe_l[l]);
            }

            hypre_DestroyPTopology(PTopology_vdata);
            
            for (i= 0; i< num_OffProcRows; i++)
            {
               hypre_MaxwellOffProcRowDestroy((void *) OffProcRows[i]);
            }
            hypre_TFree(OffProcRows);
         }

         else
         {
            Pe_l[l]= hypre_Maxwell_PNedelec(topological_edge[l],
                                            topological_edge[l+1],
                                            rfactor);
         }
#if DEBUG
#endif


         ReT_l[l]= Pe_l[l];
         hypre_BoomerAMGBuildCoarseOperator(
             (hypre_ParCSRMatrix *) hypre_IJMatrixObject(Pe_l[l]), 
                                    Aee_l[l],
             (hypre_ParCSRMatrix *) hypre_IJMatrixObject(Pe_l[l]), 
                                   &Aee_l[l+1]);

        /* zero off boundary points */
         hypre_ParCSRMatrixEliminateRowsCols(Aee_l[l+1], 
                                             BdryRanksCnts_l[l+1],
                                             BdryRanks_l[l+1]);

         hypre_ParCSRMatrixTranspose(
              (hypre_ParCSRMatrix *) hypre_IJMatrixObject(Pe_l[l]), 
                                    &transpose, 1);
         parcsr_mat= hypre_ParMatmul(transpose, Aen_l[l]);
         Aen_l[l+1]= hypre_ParMatmul(parcsr_mat, Pn_l[l]);

         hypre_ParCSRMatrixDestroy(parcsr_mat);
         hypre_ParCSRMatrixDestroy(transpose);

         xe_l[l+1] =
         hypre_ParVectorCreate(hypre_ParCSRMatrixComm(Aee_l[l+1]),
                               hypre_ParCSRMatrixGlobalNumRows(Aee_l[l+1]),
                               hypre_ParCSRMatrixRowStarts(Aee_l[l+1]));
         hypre_ParVectorInitialize(xe_l[l+1]);
         hypre_ParVectorSetPartitioningOwner(xe_l[l+1], 0);

         be_l[l+1] =
         hypre_ParVectorCreate(hypre_ParCSRMatrixComm(Aee_l[l+1]),
                               hypre_ParCSRMatrixGlobalNumRows(Aee_l[l+1]),
                               hypre_ParCSRMatrixRowStarts(Aee_l[l+1]));
         hypre_ParVectorInitialize(be_l[l+1]);
         hypre_ParVectorSetPartitioningOwner(be_l[l+1],0);

         rese_l[l+1] =
         hypre_ParVectorCreate(hypre_ParCSRMatrixComm(Aee_l[l+1]),
                               hypre_ParCSRMatrixGlobalNumRows(Aee_l[l+1]),
                               hypre_ParCSRMatrixRowStarts(Aee_l[l+1]));
         hypre_ParVectorInitialize(rese_l[l+1]);
         hypre_ParVectorSetPartitioningOwner(rese_l[l+1],0);

         ee_l[l+1] =
         hypre_ParVectorCreate(hypre_ParCSRMatrixComm(Aee_l[l+1]),
                               hypre_ParCSRMatrixGlobalNumRows(Aee_l[l+1]),
                               hypre_ParCSRMatrixRowStarts(Aee_l[l+1]));
         hypre_ParVectorInitialize(ee_l[l+1]);
         hypre_ParVectorSetPartitioningOwner(ee_l[l+1],0);

         eVtemp_l[l+1] =
         hypre_ParVectorCreate(hypre_ParCSRMatrixComm(Aee_l[l+1]),
                               hypre_ParCSRMatrixGlobalNumRows(Aee_l[l+1]),
                               hypre_ParCSRMatrixRowStarts(Aee_l[l+1]));
         hypre_ParVectorInitialize(eVtemp_l[l+1]);
         hypre_ParVectorSetPartitioningOwner(eVtemp_l[l+1],0);

         eVtemp2_l[l+1] =
         hypre_ParVectorCreate(hypre_ParCSRMatrixComm(Aee_l[l+1]),
                               hypre_ParCSRMatrixGlobalNumRows(Aee_l[l+1]),
                               hypre_ParCSRMatrixRowStarts(Aee_l[l+1]));
         hypre_ParVectorInitialize(eVtemp2_l[l+1]);
         hypre_ParVectorSetPartitioningOwner(eVtemp2_l[l+1],0);

      }  /* if (l < edge_numlevels) */
   }     /* for (l = 0; l < (en_numlevels - 1); l++) */

  /* possible to have more edge levels */
   for (l = (en_numlevels-1); l < (edge_numlevels - 1); l++)
   {
      if (!constant_coef)
      {
         void             *PTopology_vdata;
         hypre_PTopology  *PTopology;

         hypre_CreatePTopology(&PTopology_vdata);
         if (ndim > 2)
         {
            Pe_l[l]= hypre_Maxwell_PTopology(topological_edge[l],
                                             topological_edge[l+1],
                                             topological_face[l],
                                             topological_face[l+1],
                                             topological_cell[l],
                                             topological_cell[l+1],
                                             Aee_l[l],
                                             rfactor,
                                             PTopology_vdata);
         }
         else
         {
            Pe_l[l]= hypre_Maxwell_PTopology(topological_edge[l],
                                             topological_edge[l+1],
                                             topological_edge[l],
                                             topological_edge[l+1],
                                             topological_cell[l],
                                             topological_cell[l+1],
                                             Aee_l[l],
                                             rfactor,
                                             PTopology_vdata);
         }

         PTopology= PTopology_vdata;

        /* extract off-processors rows of Pe_l[l]. Needed for amge.*/
          hypre_SStructSharedDOF_ParcsrMatRowsComm(egrid_l[l],
                 (hypre_ParCSRMatrix *) hypre_IJMatrixObject(Pe_l[l]),
                                       &num_OffProcRows,
                                       &OffProcRows);
         if (ndim == 3)
         {
             hypre_ND1AMGeInterpolation(Aee_l[l],
               (hypre_ParCSRMatrix *) hypre_IJMatrixObject(PTopology -> Element_iedge),
               (hypre_ParCSRMatrix *) hypre_IJMatrixObject(PTopology -> Face_iedge),
               (hypre_ParCSRMatrix *) hypre_IJMatrixObject(PTopology -> Edge_iedge),
               (hypre_ParCSRMatrix *) hypre_IJMatrixObject(PTopology -> Element_Face),
               (hypre_ParCSRMatrix *) hypre_IJMatrixObject(PTopology -> Element_Edge),
                num_OffProcRows,
                OffProcRows,
                Pe_l[l]);
         }
         else
         {
             hypre_ND1AMGeInterpolation(Aee_l[l],
               (hypre_ParCSRMatrix *) hypre_IJMatrixObject(PTopology -> Element_iedge),
               (hypre_ParCSRMatrix *) hypre_IJMatrixObject(PTopology -> Edge_iedge),
               (hypre_ParCSRMatrix *) hypre_IJMatrixObject(PTopology -> Edge_iedge),
               (hypre_ParCSRMatrix *) hypre_IJMatrixObject(PTopology -> Element_Edge),
               (hypre_ParCSRMatrix *) hypre_IJMatrixObject(PTopology -> Element_Edge),
                num_OffProcRows,
                OffProcRows,
                Pe_l[l]);
         }

         hypre_DestroyPTopology(PTopology_vdata);
         for (i= 0; i< num_OffProcRows; i++)
         {
            hypre_MaxwellOffProcRowDestroy((void *) OffProcRows[i]);
         }
         hypre_TFree(OffProcRows);
      }

      else
      {
         Pe_l[l]= hypre_Maxwell_PNedelec(topological_edge[l],
                                         topological_edge[l+1],
                                         rfactor);
      }

      ReT_l[l]= Pe_l[l];
      hypre_BoomerAMGBuildCoarseOperator(
             (hypre_ParCSRMatrix *) hypre_IJMatrixObject(Pe_l[l]), 
                                    Aee_l[l],
             (hypre_ParCSRMatrix *) hypre_IJMatrixObject(Pe_l[l]), 
                                   &Aee_l[l+1]);

     /* zero off boundary points */
      hypre_ParCSRMatrixEliminateRowsCols(Aee_l[l+1], 
                                          BdryRanksCnts_l[l+1],
                                          BdryRanks_l[l+1]);

      xe_l[l+1] =
      hypre_ParVectorCreate(hypre_ParCSRMatrixComm(Aee_l[l+1]),
                            hypre_ParCSRMatrixGlobalNumRows(Aee_l[l+1]),
                            hypre_ParCSRMatrixRowStarts(Aee_l[l+1]));
      hypre_ParVectorInitialize(xe_l[l+1]);
      hypre_ParVectorSetPartitioningOwner(xe_l[l+1], 0);

      be_l[l+1] =
      hypre_ParVectorCreate(hypre_ParCSRMatrixComm(Aee_l[l+1]),
                            hypre_ParCSRMatrixGlobalNumRows(Aee_l[l+1]),
                            hypre_ParCSRMatrixRowStarts(Aee_l[l+1]));
      hypre_ParVectorInitialize(be_l[l+1]);
      hypre_ParVectorSetPartitioningOwner(be_l[l+1],0);

      ee_l[l+1] =
      hypre_ParVectorCreate(hypre_ParCSRMatrixComm(Aee_l[l+1]),
                            hypre_ParCSRMatrixGlobalNumRows(Aee_l[l+1]),
                            hypre_ParCSRMatrixRowStarts(Aee_l[l+1]));
      hypre_ParVectorInitialize(ee_l[l+1]);
      hypre_ParVectorSetPartitioningOwner(ee_l[l+1],0);

      rese_l[l+1] =
      hypre_ParVectorCreate(hypre_ParCSRMatrixComm(Aee_l[l+1]),
                            hypre_ParCSRMatrixGlobalNumRows(Aee_l[l+1]),
                            hypre_ParCSRMatrixRowStarts(Aee_l[l+1]));
      hypre_ParVectorInitialize(rese_l[l+1]);
      hypre_ParVectorSetPartitioningOwner(rese_l[l+1],0);

      eVtemp_l[l+1] =
      hypre_ParVectorCreate(hypre_ParCSRMatrixComm(Aee_l[l+1]),
                            hypre_ParCSRMatrixGlobalNumRows(Aee_l[l+1]),
                            hypre_ParCSRMatrixRowStarts(Aee_l[l+1]));
      hypre_ParVectorInitialize(eVtemp_l[l+1]);
      hypre_ParVectorSetPartitioningOwner(eVtemp_l[l+1],0);

      eVtemp2_l[l+1] =
      hypre_ParVectorCreate(hypre_ParCSRMatrixComm(Aee_l[l+1]),
                            hypre_ParCSRMatrixGlobalNumRows(Aee_l[l+1]),
                            hypre_ParCSRMatrixRowStarts(Aee_l[l+1]));
      hypre_ParVectorInitialize(eVtemp2_l[l+1]);
      hypre_ParVectorSetPartitioningOwner(eVtemp2_l[l+1],0);
   }

   /* Can delete all topological grids. Not even referenced in IJMatrices. */
   for (l = 0; l < edge_numlevels; l++)
   {
      HYPRE_SStructGridDestroy(topological_edge[l]);
      HYPRE_SStructGridDestroy(topological_cell[l]);
      if (ndim > 2)
      {
         HYPRE_SStructGridDestroy(topological_face[l]);
      }
   }
   hypre_TFree(topological_edge);
   hypre_TFree(topological_cell);
   if (ndim > 2)
   {
      hypre_TFree(topological_face);
   }

#if DEBUG
#endif

  (maxwell_TV_data -> Aee_l)    = Aee_l;
  (maxwell_TV_data -> Aen_l)    = Aen_l;
  (maxwell_TV_data -> Pe_l)     = Pe_l;
  (maxwell_TV_data -> ReT_l)    = ReT_l;
  (maxwell_TV_data -> xe_l)     = xe_l;
  (maxwell_TV_data -> be_l)     = be_l;
  (maxwell_TV_data -> ee_l)     = ee_l;
  (maxwell_TV_data -> rese_l)   = rese_l;
  (maxwell_TV_data -> eVtemp_l) = eVtemp_l;
  (maxwell_TV_data -> eVtemp2_l)= eVtemp2_l;

  /*-----------------------------------------------------
   * Determine relaxation parameters for edge problems.
   * Needed for quick parallel over/under-relaxation.
   *-----------------------------------------------------*/
   erelax_type  = 2;
   erelax_weight= hypre_TAlloc(double, edge_numlevels);
   eomega       = hypre_TAlloc(double, edge_numlevels);
   eCF_marker_l = hypre_TAlloc(HYPRE_Int *, edge_numlevels);

   relax_type= 6; /* SSOR */
   /*for (l= 0; l< 1; l++)
   {
      erelax_weight[l]= 1.0;
      eCF_marker_l[l]= NULL;

      e_amg_vdata= (void *) hypre_BoomerAMGCreate();
      e_amgData= e_amg_vdata;

      relax_types= hypre_CTAlloc(HYPRE_Int, 2);
      relax_types[1]= relax_type;

      amg_CF_marker= hypre_TAlloc(HYPRE_Int *, 1);
      A_array      = hypre_TAlloc(hypre_ParCSRMatrix *, 1);

      amg_CF_marker[0]= NULL;
      A_array[0]      = Aee_l[l];

     (e_amgData -> CF_marker_array)   = amg_CF_marker;
     (e_amgData -> A_array)           = A_array;
     (e_amgData -> Vtemp )            = eVtemp_l[l];
     (e_amgData -> grid_relax_type)   = relax_types;
     (e_amgData -> smooth_num_levels) = 0;
     (e_amgData -> smooth_type)       = 0;
      hypre_BoomerAMGCGRelaxWt((void *) e_amgData, 0, numCGSweeps, &eomega[l]);

       hypre_TFree((e_amgData -> A_array));
       hypre_TFree((e_amgData -> CF_marker_array));
       hypre_TFree((e_amgData -> grid_relax_type));
       (e_amgData -> A_array)= NULL;
       (e_amgData -> Vtemp ) = NULL;
       (e_amgData -> CF_marker_array)= NULL;
       (e_amgData -> grid_relax_type)= NULL;
       hypre_TFree(e_amg_vdata);
       eomega[l]= 1.0;
   }*/

   for (l= 0; l< edge_numlevels; l++)
   {
      erelax_weight[l]= 1.0;
      eomega[l]= 1.0;
      eCF_marker_l[l]= NULL;
   }
   (maxwell_TV_data ->  erelax_type)  = erelax_type;
   (maxwell_TV_data ->  erelax_weight)= erelax_weight;
   (maxwell_TV_data ->  eomega)       = eomega;
   (maxwell_TV_data ->  eCF_marker_l) = eCF_marker_l;


   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/

   if ((maxwell_TV_data -> logging) > 0)
   {
      i= (maxwell_TV_data -> max_iter);
     (maxwell_TV_data -> norms)     = hypre_TAlloc(double, i);
     (maxwell_TV_data -> rel_norms) = hypre_TAlloc(double, i);
   }

   return ierr;
}

HYPRE_Int
hypre_CoarsenPGrid( hypre_SStructGrid  *fgrid,
                    hypre_Index         index,
                    hypre_Index         stride,
                    HYPRE_Int           part,
                    hypre_SStructGrid  *cgrid,
                    HYPRE_Int          *nboxes)
{
   HYPRE_Int ierr = 0;

   hypre_SStructPGrid *pgrid= hypre_SStructGridPGrid(fgrid, part);
   hypre_StructGrid   *sgrid= hypre_SStructPGridCellSGrid(pgrid);

   hypre_BoxArray     *boxes;
   hypre_Box          *box, *contract_box;
   HYPRE_Int           i;

   /*-----------------------------------------
    * Set the coarse sgrid
    *-----------------------------------------*/
   boxes = hypre_BoxArrayDuplicate(hypre_StructGridBoxes(sgrid));
   for (i = 0; i < hypre_BoxArraySize(boxes); i++)
   {
      box = hypre_BoxArrayBox(boxes, i);

     /* contract box so that divisible by stride */
      contract_box= hypre_BoxContraction(box, sgrid, stride);
      hypre_ProjectBox(contract_box, index, stride);

      hypre_StructMapFineToCoarse(hypre_BoxIMin(contract_box), index, stride,
                                  hypre_BoxIMin(contract_box));
      hypre_StructMapFineToCoarse(hypre_BoxIMax(contract_box), index, stride,
                                  hypre_BoxIMax(contract_box));

     /* set box even if zero volume but don't count it */
      HYPRE_SStructGridSetExtents(cgrid, part, 
                                  hypre_BoxIMin(contract_box), 
                                  hypre_BoxIMax(contract_box));

      if ( hypre_BoxVolume(contract_box) )
      {
         *nboxes= *nboxes+1;
      }
      hypre_BoxDestroy(contract_box);
   }
   hypre_BoxArrayDestroy(boxes);

   return ierr;
}



/*--------------------------------------------------------------------------
 *  Contracts a box so that the resulting box divides evenly into rfactor.
 *  Contraction is done in the (+) or (-) direction that does not have
 *  neighbor boxes, or if both directions have neighbor boxes, the (-) side
 *  is contracted.
 *  Modified to use box manager AHB 11/06
 *--------------------------------------------------------------------------*/

hypre_Box *
hypre_BoxContraction( hypre_Box           *box,
                      hypre_StructGrid    *sgrid,
                      hypre_Index          rfactor )
{

   hypre_BoxManager    *boxman = hypre_StructGridBoxMan(sgrid);

   hypre_BoxArray      *neighbor_boxes= NULL;
   hypre_Box           *nbox;
   hypre_Box           *contracted_box;
   hypre_Box           *shifted_box;
   hypre_Box            intersect_box;

   HYPRE_Int            ndim= hypre_StructGridDim(sgrid);

   hypre_Index          remainder, box_width;
   HYPRE_Int            i, j, k, p;
   HYPRE_Int            npos, nneg;


   /* get the boxes out of the box manager - use these as the neighbor boxes */
   neighbor_boxes = hypre_BoxArrayCreate(0);
   hypre_BoxManGetAllEntriesBoxes( boxman, neighbor_boxes);


   contracted_box= hypre_BoxCreate();

   hypre_ClearIndex(remainder);
   p= 0;
   for (i= 0; i< ndim; i++)
   {
      j= hypre_BoxIMax(box)[i] - hypre_BoxIMin(box)[i] + 1;
      box_width[i]= j;
      k= j%rfactor[i];

      if (k)
      {
         remainder[i]= k;
         p++;
      }
   }

   hypre_CopyBox(box, contracted_box);
   if (p)
   {
      shifted_box= hypre_BoxCreate();
      for (i= 0; i< ndim; i++)
      {
         if (remainder[i])   /* non-divisible in the i'th direction */
         {
           /* shift box in + & - directions to determine which side to
              contract. */
            hypre_CopyBox(box, shifted_box);
            hypre_BoxIMax(shifted_box)[i]+= box_width[i];
            hypre_BoxIMin(shifted_box)[i]+= box_width[i];

            npos= 0;
            hypre_ForBoxI(k, neighbor_boxes)
            {
               nbox= hypre_BoxArrayBox(neighbor_boxes, k);
               hypre_IntersectBoxes(shifted_box, nbox, &intersect_box); 
               if (hypre_BoxVolume(&intersect_box))
               {
                  npos++;
               }
            }

            hypre_CopyBox(box, shifted_box);
            hypre_BoxIMax(shifted_box)[i]-= box_width[i];
            hypre_BoxIMin(shifted_box)[i]-= box_width[i];

            nneg= 0;
            hypre_ForBoxI(k, neighbor_boxes)
            {
               nbox= hypre_BoxArrayBox(neighbor_boxes, k);
               hypre_IntersectBoxes(shifted_box, nbox, &intersect_box);
               if (hypre_BoxVolume(&intersect_box))
               {
                  nneg++;
               }
            }

            if ( (npos) || ( (!npos) && (!nneg) ) )
            {
               /* contract - direction */
                hypre_BoxIMin(contracted_box)[i]+= remainder[i];
            }
            else
            {
               if (nneg)
               {
                  /* contract + direction */
                   hypre_BoxIMax(contracted_box)[i]-= remainder[i];
               }
            }

         }  /* if (remainder[i]) */
      }     /* for (i= 0; i< ndim; i++) */

      hypre_BoxDestroy(shifted_box);
   }  /* if (p) */
             
   hypre_BoxArrayDestroy(neighbor_boxes);

   return contracted_box;
}


