/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.10 $
 ***********************************************************************EHEADER*/

/******************************************************************************
 * OpenMP Problems
 *
 * Need to fix the way these variables are set and incremented in loops:
 *   nElements, nElements_iedges, nFaces, nFaces_iedges, nEdges, nEdges_iedges,
 *   nElements_Faces, nElements_Edges,
 *   j, l, k (these three only where they are listed at the end of SMP_PRIVATE)
 *
 ******************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_Maxwell_Interp.c
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_CreatePTopology(void **PTopology_vdata_ptr)
{
   hypre_PTopology   *PTopology;
   HYPRE_Int          ierr= 0;

   PTopology= hypre_CTAlloc(hypre_PTopology, 1);

   (PTopology ->  Face_iedge)   = NULL;
   (PTopology ->  Element_iedge)= NULL;
   (PTopology ->  Edge_iedge)   = NULL;

   (PTopology ->  Element_Face) = NULL;
   (PTopology ->  Element_Edge) = NULL;

   *PTopology_vdata_ptr= (void *) PTopology;

   return ierr;
}

HYPRE_Int
hypre_DestroyPTopology(void *PTopology_vdata)
{
   hypre_PTopology       *PTopology= PTopology_vdata;
   HYPRE_Int              ierr     = 0;

   if (PTopology)
   {
      if ( (PTopology -> Face_iedge) != NULL)
      {
         HYPRE_IJMatrixDestroy(PTopology -> Face_iedge);
      }
      HYPRE_IJMatrixDestroy(PTopology -> Element_iedge);
      HYPRE_IJMatrixDestroy(PTopology -> Edge_iedge);

      if ( (PTopology -> Element_Face) != NULL)
      {
         HYPRE_IJMatrixDestroy(PTopology -> Element_Face);
      }
      HYPRE_IJMatrixDestroy(PTopology -> Element_Edge);
   }
   hypre_TFree(PTopology);

   return ierr;
}

hypre_IJMatrix *
hypre_Maxwell_PTopology(  hypre_SStructGrid    *fgrid_edge,
                          hypre_SStructGrid    *cgrid_edge,
                          hypre_SStructGrid    *fgrid_face,
                          hypre_SStructGrid    *cgrid_face,
                          hypre_SStructGrid    *fgrid_element,
                          hypre_SStructGrid    *cgrid_element,
                          hypre_ParCSRMatrix   *Aee,
                          hypre_Index           rfactor,       
                          void                 *PTopology_vdata)
{
   MPI_Comm               comm = (fgrid_element ->  comm);

   hypre_PTopology       *PTopology= PTopology_vdata;

   hypre_IJMatrix        *Face_iedge;
   hypre_IJMatrix        *Element_iedge;
   hypre_IJMatrix        *Edge_iedge;

   hypre_IJMatrix        *Element_Face;
   hypre_IJMatrix        *Element_Edge;

   hypre_IJMatrix        *edge_Edge;

   hypre_SStructPGrid    *p_cgrid, *p_fgrid;
   hypre_StructGrid      *var_cgrid,  *var_fgrid;
   hypre_BoxArray        *cboxes, *fboxes, *box_array;
   hypre_Box             *cbox, *fbox, *cellbox, *vbox, copy_box;

   hypre_BoxArray       **contract_fedgeBoxes;
   hypre_Index          **Edge_cstarts, **upper_shifts, **lower_shifts;
   HYPRE_Int            **cfbox_mapping, **fcbox_mapping;

   hypre_BoxManEntry     *entry;
   HYPRE_Int              rank, rank2;
   HYPRE_Int              start_rank1; 

   HYPRE_Int              nFaces, nEdges, nElements, nedges;
   HYPRE_Int              nxFaces, nyFaces, nzFaces, nxEdges, nyEdges, nzEdges;
   HYPRE_Int              n_xFace_iedges, n_yFace_iedges, n_zFace_iedges;
   HYPRE_Int              n_Cell_iedges;

   HYPRE_Int              nElements_iedges, nFaces_iedges, nEdges_iedges;
   HYPRE_Int              nElements_Faces, nElements_Edges, nEdges_edges;

   HYPRE_Int             *iFace, *iEdge, *iElement, *iedgeEdge;
   HYPRE_Int             *jFace_edge, *jEdge_iedge, *jElement_edge;
   HYPRE_Int             *jElement_Face, *jElement_Edge, *jedge_Edge;

   double                *vals_ElementEdge, *vals_ElementFace, *vals_edgeEdge, *vals_Faceedge;
   double                *vals_Elementedge, *vals_Edgeiedge;
   HYPRE_Int             *ncols_Elementedge, *ncols_Edgeiedge, *ncols_edgeEdge, *ncols_Faceedge;
   HYPRE_Int             *ncols_ElementFace, *ncols_ElementEdge;
   HYPRE_Int             *bdryedge_location;
   double                 fCedge_ratio;
   double                *stencil_vals, *upper, *lower, *diag, *face_w1, *face_w2;
   HYPRE_Int             *off_proc_flag;

   hypre_Index            cindex;
   hypre_Index            findex;
   hypre_Index            var_index, cell_index, *boxoffset, *suboffset;
   hypre_Index            loop_size, start, cstart, stride, low_index, hi_index;
   hypre_Index            ishift, jshift, kshift, zero_index, one_index;
   HYPRE_Int              loopi, loopj, loopk;
   HYPRE_Int              n_boxoffsets, component_stride;

   HYPRE_Int              nparts= hypre_SStructGridNParts(fgrid_element);
   HYPRE_Int              ndim  = hypre_SStructGridNDim(fgrid_element);

   HYPRE_SStructVariable *vartypes, *Edge_vartypes, *Face_vartypes;
   hypre_Index           *varoffsets;
   HYPRE_Int             *vartype_map;
   HYPRE_Int              matrix_type= HYPRE_PARCSR;

   HYPRE_Int              nvars, Face_nvars, Edge_nvars, part, var, box, fboxi;
   HYPRE_Int              tot_vars= 8;

   HYPRE_Int              t, i, j, k, l, m, n, p, r;

   HYPRE_Int              ilower, iupper;
   HYPRE_Int              jlower, jupper;
   HYPRE_Int            **flower_ranks, **fupper_ranks;
   HYPRE_Int            **clower_ranks, **cupper_ranks;
   HYPRE_Int           ***n_CtoVbox, ****CtoVboxnums;
   HYPRE_Int             *num_vboxes, **vboxnums;

   HYPRE_Int              size1;
   HYPRE_Int              true = 1;
   HYPRE_Int              false= 0;
   HYPRE_Int              row_in;

   HYPRE_Int              myproc;

   hypre_MPI_Comm_rank(comm, &myproc);
   hypre_SetIndex(ishift, 1, 0, 0);
   hypre_SetIndex(jshift, 0, 1, 0);
   hypre_SetIndex(kshift, 0, 0, 1);
   hypre_ClearIndex(zero_index);
   hypre_ClearIndex(one_index);
   for (i= 0; i< ndim; i++)
   {
      one_index[i]= 1;
   }

   /* set rfactor[2]= 1 if ndim=2. */
   if (ndim == 2)
   {
      rfactor[2]= 1;
   }

   /*-------------------------------------------------------------------
    * Find the coarse-fine connection pattern, i.e., the topology
    * needed to create the interpolation operators.
    * Face_iedge, Edge_iedge, Element_iedge, Element_Face, Element_Edge,
    * and edge_Edge connections are defined in terms of parcsr_matrices.
    * These connections are determined using the cell-centred grids.
    * Note that we are assuming the variable type enumeration 
    * given in hypre_SStructVariable_enum. 
    *
    * We consider both 2-d and 3-d cases. In 2-d, the edges are faces.
    * We will continue to call them edges, but use the face variable
    * enumeration.
    *-------------------------------------------------------------------*/
   varoffsets= hypre_CTAlloc(hypre_Index, tot_vars);
   
   /* total of 8 variable types. Create a mapping between user enumeration
      to hypre enumeration. Only need for face and edge grids. */
   vartype_map= hypre_CTAlloc(HYPRE_Int, 8);

   part= 0;
   p_cgrid = hypre_SStructGridPGrid(cgrid_face, part);   /* face cgrid */
   nvars   = hypre_SStructPGridNVars(p_cgrid);
   vartypes= hypre_SStructPGridVarTypes(p_cgrid);

   for (i= 0; i< nvars; i++)
   {
      t= vartypes[i];
      switch(t)
      {
         case 2:
         {
            vartype_map[2]= i;
            break;
         }
        
         case 3:
         {
            vartype_map[3]= i;
            break;
         }
        
         case 4:
         {
            vartype_map[4]= i;
            break;
         }
      }
   }
        
   if (ndim == 3)
   {
      p_cgrid = hypre_SStructGridPGrid(cgrid_edge, part);   /* edge cgrid */
      nvars   = hypre_SStructPGridNVars(p_cgrid);
      vartypes= hypre_SStructPGridVarTypes(p_cgrid);

      for (i= 0; i< nvars; i++)
      {
         t= vartypes[i];
         switch(t)
         {
            case 5:
            {
               vartype_map[5]= i;
               break;
            }
        
            case 6:
            {
               vartype_map[6]= i;
               break;
            }
        
            case 7:
            {
               vartype_map[7]= i;
               break;
            }
         }
      }
   }

   /* local sizes */
   nFaces   = 0;
   nEdges   = 0;
   nElements= 0;
   nedges   = 0;

   nxFaces  = 0;
   nyFaces  = 0;
   nzFaces  = 0;
   nxEdges  = 0;
   nyEdges  = 0;
   nzEdges  = 0;

   for (part= 0; part< nparts; part++)
   {
      p_cgrid   = hypre_SStructGridPGrid(cgrid_element, part);  /* cell cgrid */
      var_cgrid = hypre_SStructPGridCellSGrid(p_cgrid) ;
      nElements+= hypre_StructGridLocalSize(var_cgrid);

      t= 0;
      hypre_SStructVariableGetOffset((hypre_SStructVariable) t,
                                     ndim, varoffsets[0]);

      p_cgrid = hypre_SStructGridPGrid(cgrid_face, part);       /* face cgrid */
      nvars   = hypre_SStructPGridNVars(p_cgrid);
      vartypes= hypre_SStructPGridVarTypes(p_cgrid);

      for (var= 0; var< nvars; var++)
      {
         var_cgrid= hypre_SStructPGridSGrid(p_cgrid, var);
         t= vartypes[var];
         nFaces+= hypre_StructGridLocalSize(var_cgrid);

         switch(t)
         {
            case 2:
               nxFaces+= hypre_StructGridLocalSize(var_cgrid);
               break;
            case 3:
               nyFaces+= hypre_StructGridLocalSize(var_cgrid);
               break;
            case 4:
               nzFaces+= hypre_StructGridLocalSize(var_cgrid);
               break;
         }

         hypre_SStructVariableGetOffset((hypre_SStructVariable) t,
                                        ndim, varoffsets[t]);
      }

      /* 2-d vs 3-d case */
      if (ndim < 3)
      {
         nEdges = nFaces;
         nxEdges= nxFaces;
         nyEdges= nyFaces;
         nzEdges= 0;
      }

      else
      {
         p_cgrid = hypre_SStructGridPGrid(cgrid_edge, part);    /* edge cgrid */
         nvars   = hypre_SStructPGridNVars(p_cgrid);
         vartypes= hypre_SStructPGridVarTypes(p_cgrid);

         for (var= 0; var< nvars; var++)
         {
            var_cgrid= hypre_SStructPGridSGrid(p_cgrid, var);
            t= vartypes[var];
            nEdges+= hypre_StructGridLocalSize(var_cgrid);

            switch(t)
            {
               case 5:
                  nxEdges+= hypre_StructGridLocalSize(var_cgrid);
                  break;
               case 6:
                  nyEdges+= hypre_StructGridLocalSize(var_cgrid);
                  break;
               case 7:
                  nzEdges+= hypre_StructGridLocalSize(var_cgrid);
                  break;
            }

            hypre_SStructVariableGetOffset((hypre_SStructVariable) t,
                                           ndim, varoffsets[t]);
         }
      }

      /* same for 2-d & 3-d, assuming that fgrid_edge= fgrid_face in input */
      p_fgrid = hypre_SStructGridPGrid(fgrid_edge, part);    /* edge fgrid */
      nvars   = hypre_SStructPGridNVars(p_fgrid);
      vartypes= hypre_SStructPGridVarTypes(p_fgrid);

      for (var= 0; var< nvars; var++)
      {
         var_fgrid= hypre_SStructPGridSGrid(p_fgrid, var);
         nedges  += hypre_StructGridLocalSize(var_fgrid);
      }
   }

   /*--------------------------------------------------------------------------
    *  Form mappings between the c & f box numbers. Note that a cbox
    *  can land inside only one fbox since the latter was contracted. Without
    *  the extraction, a cbox can land in more than 1 fboxes (e.g., cbox
    *  boundary extending into other fboxes). These mappings are for the 
    *  cell-centred boxes. 
    *  Check: Other variable boxes should follow this mapping, by
    *  property of the variable-shifted indices? Can the cell-centred boundary
    *  indices of a box be non-cell-centred indices for another box?
    * 
    *  Also determine contracted cell-centred fboxes.
    *--------------------------------------------------------------------------*/
   cfbox_mapping= hypre_TAlloc(HYPRE_Int *, nparts);
   fcbox_mapping= hypre_TAlloc(HYPRE_Int *, nparts);
   contract_fedgeBoxes= hypre_TAlloc(hypre_BoxArray *, nparts);
   Edge_cstarts= hypre_TAlloc(hypre_Index *, nparts);
   upper_shifts= hypre_TAlloc(hypre_Index *, nparts);
   lower_shifts= hypre_TAlloc(hypre_Index *, nparts);

   for (i= 0; i< nparts; i++)
   {
      p_fgrid  = hypre_SStructGridPGrid(fgrid_element, i);
      var_fgrid= hypre_SStructPGridCellSGrid(p_fgrid);
      fboxes   = hypre_StructGridBoxes(var_fgrid);
      j        = hypre_BoxArraySize(fboxes);
      fcbox_mapping[i]= hypre_CTAlloc(HYPRE_Int, j);

      p_cgrid  = hypre_SStructGridPGrid(cgrid_element, i);
      var_cgrid= hypre_SStructPGridCellSGrid(p_cgrid);
      cboxes   = hypre_StructGridBoxes(var_cgrid);
      j        = hypre_BoxArraySize(fboxes);
      cfbox_mapping[i]= hypre_CTAlloc(HYPRE_Int, j);

      /* assuming if i1 > i2 and (box j1) is coarsened from (box i1)
         and (box j2) from (box i2), then j1 > j2. */
      k= 0;
      hypre_ForBoxI(j, fboxes)
      {
         fbox= hypre_BoxArrayBox(fboxes, j);
         hypre_CopyBox(fbox, &copy_box);
         hypre_ProjectBox(&copy_box, zero_index, rfactor);
         hypre_StructMapFineToCoarse(hypre_BoxIMin(&copy_box), zero_index,
                                     rfactor, hypre_BoxIMin(&copy_box));
         hypre_StructMapFineToCoarse(hypre_BoxIMax(&copy_box), zero_index,
                                     rfactor, hypre_BoxIMax(&copy_box));

         /* since the ordering of the cboxes was determined by the fbox
            ordering, we only have to check if the first cbox in the
            list intersects with copy_box. If not, this fbox vanished in the
            coarsening. Note that this gives you the correct interior cbox. */
         cbox= hypre_BoxArrayBox(cboxes, k);
         hypre_IntersectBoxes(&copy_box, cbox, &copy_box);
         if (hypre_BoxVolume(&copy_box))
         {
            cfbox_mapping[i][k]= j;
            fcbox_mapping[i][j]= k;
            k++;
         }  /* if (hypre_BoxVolume(&copy_box)) */
      }     /* hypre_ForBoxI(j, fboxes) */

      /* fill up the contracted box_array */
      contract_fedgeBoxes[i]= hypre_BoxArrayCreate(0);
      Edge_cstarts[i]= hypre_TAlloc(hypre_Index, hypre_BoxArraySize(fboxes));
      upper_shifts[i]= hypre_TAlloc(hypre_Index, hypre_BoxArraySize(fboxes));
      lower_shifts[i]= hypre_TAlloc(hypre_Index, hypre_BoxArraySize(fboxes));
      hypre_ForBoxI(j, fboxes)
      {
         fbox= hypre_BoxArrayBox(fboxes, j);

         /* contract the fbox to correspond to the correct cbox */
         cbox= hypre_BoxContraction(fbox, var_fgrid, rfactor);
         hypre_AppendBox(cbox, contract_fedgeBoxes[i]);

         /* record the offset mapping between the coarse cell index and
            the fine cell index */
         hypre_ClearIndex(upper_shifts[i][j]);
         hypre_ClearIndex(lower_shifts[i][j]);
         for (l= 0; l< ndim; l++)
         {
            m= hypre_BoxIMin(cbox)[l];
            p= m%rfactor[l];
            if (p > 0 && m > 0)
            {
               upper_shifts[i][j][l]= p-1;
               lower_shifts[i][j][l]= p-rfactor[l];
            }
            else
            {
               upper_shifts[i][j][l]= rfactor[l]-p-1;
               lower_shifts[i][j][l]=-p;
            }
         }

         /* record the cstarts of the cbox */
         hypre_ProjectBox(cbox, zero_index, rfactor);
         hypre_CopyIndex(hypre_BoxIMin(cbox), Edge_cstarts[i][j]);
         hypre_StructMapFineToCoarse(Edge_cstarts[i][j], zero_index, rfactor,
                                     Edge_cstarts[i][j]);

         hypre_BoxDestroy(cbox);
      }
   }  /* for (i= 0; i< nparts; i++) */
 
   /* variable rank bounds for this processor */
   n_CtoVbox   = hypre_TAlloc(HYPRE_Int **, nparts);
   CtoVboxnums = hypre_TAlloc(HYPRE_Int ***, nparts);
   for (part= 0; part< nparts; part++)
   {
      hypre_SStructCellGridBoxNumMap(fgrid_edge, part, &n_CtoVbox[part],
                                     &CtoVboxnums[part]);
   }

   /* variable rank bounds for this processor */
   flower_ranks= hypre_TAlloc(HYPRE_Int *, nparts);
   fupper_ranks= hypre_TAlloc(HYPRE_Int *, nparts);

   clower_ranks= hypre_TAlloc(HYPRE_Int *, nparts);
   cupper_ranks= hypre_TAlloc(HYPRE_Int *, nparts);
   for (part= 0; part< nparts; part++)
   {
      flower_ranks[part]= hypre_CTAlloc(HYPRE_Int, tot_vars);
      fupper_ranks[part]= hypre_CTAlloc(HYPRE_Int, tot_vars);

      /* cell grid ranks */
      p_fgrid= hypre_SStructGridPGrid(fgrid_element, part);
      var_fgrid= hypre_SStructPGridSGrid(p_fgrid, 0);
      box_array= hypre_StructGridBoxes(var_fgrid);

      fbox     = hypre_BoxArrayBox(box_array, 0);
      hypre_CopyIndex(hypre_BoxIMin(fbox), findex);
      hypre_SStructGridFindBoxManEntry(fgrid_element, part, findex, 0,
                                       &entry);
      hypre_SStructBoxManEntryGetGlobalRank(entry, findex, &flower_ranks[part][0],
                                            matrix_type);

      fbox= hypre_BoxArrayBox(box_array, hypre_BoxArraySize(box_array)-1);
      hypre_CopyIndex(hypre_BoxIMax(fbox), findex);
      hypre_SStructGridFindBoxManEntry(fgrid_edge, part, findex, 0,
                                       &entry);
      hypre_SStructBoxManEntryGetGlobalRank(entry, findex, &fupper_ranks[part][0],
                                            matrix_type);

      clower_ranks[part]= hypre_CTAlloc(HYPRE_Int, tot_vars);
      cupper_ranks[part]= hypre_CTAlloc(HYPRE_Int, tot_vars);

      p_cgrid= hypre_SStructGridPGrid(cgrid_element, part);
      var_cgrid= hypre_SStructPGridSGrid(p_cgrid, 0);
      box_array= hypre_StructGridBoxes(var_cgrid);

      cbox     = hypre_BoxArrayBox(box_array, 0);
      hypre_CopyIndex(hypre_BoxIMin(cbox), cindex);
      hypre_SStructGridFindBoxManEntry(cgrid_element, part, cindex, 0,
                                       &entry);
      hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &clower_ranks[part][0],
                                            matrix_type);

      cbox= hypre_BoxArrayBox(box_array, hypre_BoxArraySize(box_array)-1);
      hypre_CopyIndex(hypre_BoxIMax(cbox), cindex);
      hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex, 0,
                                       &entry);
      hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &cupper_ranks[part][0],
                                            matrix_type);

      /* face grid ranks */
      p_fgrid= hypre_SStructGridPGrid(fgrid_face, part);
      p_cgrid= hypre_SStructGridPGrid(cgrid_face, part);
      nvars  = hypre_SStructPGridNVars(p_fgrid);
      vartypes= hypre_SStructPGridVarTypes(p_cgrid);

      for (i= 0; i< nvars; i++)
      {
         t= vartypes[i];
         var_fgrid= hypre_SStructPGridSGrid(p_fgrid, i);
         box_array= hypre_StructGridBoxes(var_fgrid);

         fbox     = hypre_BoxArrayBox(box_array, 0);
         hypre_CopyIndex(hypre_BoxIMin(fbox), findex);
         hypre_SStructGridFindBoxManEntry(fgrid_face, part, findex, i,
                                          &entry);
         hypre_SStructBoxManEntryGetGlobalRank(entry, findex, &flower_ranks[part][t],
                                               matrix_type);

         fbox= hypre_BoxArrayBox(box_array, hypre_BoxArraySize(box_array)-1);
         hypre_CopyIndex(hypre_BoxIMax(fbox), findex);
         hypre_SStructGridFindBoxManEntry(fgrid_face, part, findex, i,
                                          &entry);
         hypre_SStructBoxManEntryGetGlobalRank(entry, findex, &fupper_ranks[part][t],
                                               matrix_type);

         var_cgrid= hypre_SStructPGridSGrid(p_cgrid, i);
         box_array= hypre_StructGridBoxes(var_cgrid);
         cbox     = hypre_BoxArrayBox(box_array, 0);
         hypre_CopyIndex(hypre_BoxIMin(cbox), cindex);
         hypre_SStructGridFindBoxManEntry(cgrid_face, part, cindex, i,
                                          &entry);
         hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &clower_ranks[part][t],
                                               matrix_type);

         cbox= hypre_BoxArrayBox(box_array, hypre_BoxArraySize(box_array)-1);
         hypre_CopyIndex(hypre_BoxIMax(cbox), cindex);
         hypre_SStructGridFindBoxManEntry(cgrid_face, part, cindex, i,
                                          &entry);
         hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &cupper_ranks[part][t],
                                               matrix_type);
      }
      /* edge grid ranks */
      p_fgrid= hypre_SStructGridPGrid(fgrid_edge, part);
      p_cgrid= hypre_SStructGridPGrid(cgrid_edge, part);
      nvars  = hypre_SStructPGridNVars(p_fgrid);
      vartypes= hypre_SStructPGridVarTypes(p_cgrid);

      for (i= 0; i< nvars; i++)
      {
         t= vartypes[i];
         var_fgrid= hypre_SStructPGridSGrid(p_fgrid, i);
         box_array= hypre_StructGridBoxes(var_fgrid);

         fbox     = hypre_BoxArrayBox(box_array, 0);
         hypre_CopyIndex(hypre_BoxIMin(fbox), findex);
         hypre_SStructGridFindBoxManEntry(fgrid_edge, part, findex, i,
                                          &entry);
         hypre_SStructBoxManEntryGetGlobalRank(entry, findex, &flower_ranks[part][t],
                                               matrix_type);

         fbox= hypre_BoxArrayBox(box_array, hypre_BoxArraySize(box_array)-1);
         hypre_CopyIndex(hypre_BoxIMax(fbox), findex);
         hypre_SStructGridFindBoxManEntry(fgrid_edge, part, findex, i,
                                          &entry);
         hypre_SStructBoxManEntryGetGlobalRank(entry, findex, &fupper_ranks[part][t],
                                               matrix_type);

         var_cgrid= hypre_SStructPGridSGrid(p_cgrid, i);
         box_array= hypre_StructGridBoxes(var_cgrid);
         cbox     = hypre_BoxArrayBox(box_array, 0);
         hypre_CopyIndex(hypre_BoxIMin(cbox), cindex);
         hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex, i,
                                          &entry);
         hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &clower_ranks[part][t],
                                               matrix_type);

         cbox= hypre_BoxArrayBox(box_array, hypre_BoxArraySize(box_array)-1);
         hypre_CopyIndex(hypre_BoxIMax(cbox), cindex);
         hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex, i,
                                          &entry);
         hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &cupper_ranks[part][t],
                                               matrix_type);
      }
   }

   /* CREATE IJ_MATRICES- need to find the size of each one. Notice that the row
      and col ranks of these matrices can be created using only grid information. 
      Grab the first part, first variable, first box, and lower index (lower rank);
      Grab the last part, last variable, last box, and upper index (upper rank). */
 
   /* Element_iedge- same for 2-d and 3-d */
   /* lower rank */
   part= 0;
   box = 0;
   hypre_SStructGridBoxProcFindBoxManEntry(cgrid_element, part, 0, box, myproc, &entry);

   p_cgrid  = hypre_SStructGridPGrid(cgrid_element, part);
   var_cgrid= hypre_SStructPGridCellSGrid(p_cgrid) ;
   cboxes   = hypre_StructGridBoxes(var_cgrid);
   cbox     = hypre_BoxArrayBox(cboxes, 0);
   hypre_SStructBoxManEntryGetGlobalCSRank(entry, hypre_BoxIMin(cbox), &ilower);

   p_fgrid = hypre_SStructGridPGrid(fgrid_edge, part);
   hypre_SStructGridBoxProcFindBoxManEntry(fgrid_edge, part, 0, box, myproc, &entry);
   
   var_fgrid= hypre_SStructPGridSGrid(p_fgrid, 0);
   fboxes   = hypre_StructGridBoxes(var_fgrid);
   fbox     = hypre_BoxArrayBox(fboxes, 0);
   hypre_SStructBoxManEntryGetGlobalCSRank(entry, hypre_BoxIMin(fbox), &jlower);
   
   /* upper rank */
   part= nparts-1;
   p_cgrid  = hypre_SStructGridPGrid(cgrid_element, part);
   var_cgrid= hypre_SStructPGridCellSGrid(p_cgrid) ;
   cboxes   = hypre_StructGridBoxes(var_cgrid);
   cbox     = hypre_BoxArrayBox(cboxes, hypre_BoxArraySize(cboxes)-1);

   hypre_SStructGridBoxProcFindBoxManEntry(cgrid_element, part, 0, hypre_BoxArraySize(cboxes)-1, 
                                           myproc, &entry);
   hypre_SStructBoxManEntryGetGlobalCSRank(entry, hypre_BoxIMax(cbox), &iupper);

   p_fgrid = hypre_SStructGridPGrid(fgrid_edge, part);
   nvars   = hypre_SStructPGridNVars(p_fgrid);

   var_fgrid= hypre_SStructPGridSGrid(p_fgrid, nvars-1);
   fboxes   = hypre_StructGridBoxes(var_fgrid);
   fbox     = hypre_BoxArrayBox(fboxes, hypre_BoxArraySize(fboxes)-1);

   hypre_SStructGridBoxProcFindBoxManEntry(fgrid_edge, part, nvars-1, hypre_BoxArraySize(fboxes)-1, 
                                           myproc, &entry);
   hypre_SStructBoxManEntryGetGlobalCSRank(entry, hypre_BoxIMax(fbox), &jupper);

   HYPRE_IJMatrixCreate(comm, ilower, iupper, jlower, jupper, &Element_iedge);
   HYPRE_IJMatrixSetObjectType(Element_iedge, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(Element_iedge);

   /* Edge_iedge. Note that even though not all the iedges are involved (e.g.,
    * truncated edges are not), we use the ranks determined by the Edge/edge grids.
    * Same for 2-d and 3-d. */
   /* lower rank */
   part= 0;
   box = 0;
   hypre_SStructGridBoxProcFindBoxManEntry(cgrid_edge, part, 0, box, myproc, &entry);
   p_cgrid = hypre_SStructGridPGrid(cgrid_edge, part);
   var_cgrid= hypre_SStructPGridSGrid(p_cgrid, 0);
   cboxes   = hypre_StructGridBoxes(var_cgrid);
   cbox    = hypre_BoxArrayBox(cboxes, 0);
   hypre_SStructBoxManEntryGetGlobalCSRank(entry, hypre_BoxIMin(cbox), &ilower);

   hypre_SStructGridBoxProcFindBoxManEntry(fgrid_edge, part, 0, box, myproc, &entry);
   p_fgrid = hypre_SStructGridPGrid(fgrid_edge, part);
   var_fgrid= hypre_SStructPGridSGrid(p_fgrid, 0);
   fboxes   = hypre_StructGridBoxes(var_fgrid);
   fbox    = hypre_BoxArrayBox(fboxes, 0);
   hypre_SStructBoxManEntryGetGlobalCSRank(entry, hypre_BoxIMin(fbox), &jlower);

   /* upper rank */
   part= nparts-1;
   p_cgrid = hypre_SStructGridPGrid(cgrid_edge, part);
   nvars   = hypre_SStructPGridNVars(p_cgrid);
   var_cgrid= hypre_SStructPGridSGrid(p_cgrid, nvars-1);
   cboxes   = hypre_StructGridBoxes(var_cgrid);
   cbox    = hypre_BoxArrayBox(cboxes, hypre_BoxArraySize(cboxes)-1);
   hypre_SStructGridBoxProcFindBoxManEntry(cgrid_edge, part, nvars-1,
                                           hypre_BoxArraySize(cboxes)-1, myproc, &entry);
   hypre_SStructBoxManEntryGetGlobalCSRank(entry, hypre_BoxIMax(cbox), &iupper);

   p_fgrid = hypre_SStructGridPGrid(fgrid_edge, part);
   nvars   = hypre_SStructPGridNVars(p_fgrid);
   var_fgrid= hypre_SStructPGridSGrid(p_fgrid, nvars-1);
   fboxes   = hypre_StructGridBoxes(var_fgrid);
   fbox    = hypre_BoxArrayBox(fboxes, hypre_BoxArraySize(fboxes)-1);
   hypre_SStructGridBoxProcFindBoxManEntry(fgrid_edge, part, nvars-1,
                                           hypre_BoxArraySize(fboxes)-1, myproc, &entry);
   hypre_SStructBoxManEntryGetGlobalCSRank(entry, hypre_BoxIMax(fbox), &jupper);

   HYPRE_IJMatrixCreate(comm, ilower, iupper, jlower, jupper, &Edge_iedge);
   HYPRE_IJMatrixSetObjectType(Edge_iedge, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(Edge_iedge);

   /* edge_Edge. Same for 2-d and 3-d. */
   /* lower rank */
   part= 0;
   box = 0;
   hypre_SStructGridBoxProcFindBoxManEntry(fgrid_edge, part, 0, box, myproc, &entry);
   p_fgrid = hypre_SStructGridPGrid(fgrid_edge, part);
   var_fgrid= hypre_SStructPGridSGrid(p_fgrid, 0);
   fboxes   = hypre_StructGridBoxes(var_fgrid);
   fbox    = hypre_BoxArrayBox(fboxes, 0);
   hypre_SStructBoxManEntryGetGlobalCSRank(entry, hypre_BoxIMin(fbox), &ilower);

   hypre_SStructGridBoxProcFindBoxManEntry(cgrid_edge, part, 0, box, myproc, &entry);
   p_cgrid = hypre_SStructGridPGrid(cgrid_edge, part);
   var_cgrid= hypre_SStructPGridSGrid(p_cgrid, 0);
   cboxes   = hypre_StructGridBoxes(var_cgrid);
   cbox    = hypre_BoxArrayBox(cboxes, 0);
   hypre_SStructBoxManEntryGetGlobalCSRank(entry, hypre_BoxIMin(cbox), &jlower);

   /* upper rank */
   part= nparts-1;
   p_fgrid = hypre_SStructGridPGrid(fgrid_edge, part);
   nvars   = hypre_SStructPGridNVars(p_fgrid);
   var_fgrid= hypre_SStructPGridSGrid(p_fgrid, nvars-1);
   fboxes   = hypre_StructGridBoxes(var_fgrid);
   fbox    = hypre_BoxArrayBox(fboxes, hypre_BoxArraySize(fboxes)-1);

   hypre_SStructGridBoxProcFindBoxManEntry(fgrid_edge, part, nvars-1, 
                                           hypre_BoxArraySize(fboxes)-1, myproc, &entry);
   hypre_SStructBoxManEntryGetGlobalCSRank(entry, hypre_BoxIMax(fbox), &iupper);

   p_cgrid = hypre_SStructGridPGrid(cgrid_edge, part);
   nvars   = hypre_SStructPGridNVars(p_cgrid);
   var_cgrid= hypre_SStructPGridSGrid(p_cgrid, nvars-1);
   cboxes   = hypre_StructGridBoxes(var_cgrid);
   cbox    = hypre_BoxArrayBox(cboxes, hypre_BoxArraySize(cboxes)-1);

   hypre_SStructGridBoxProcFindBoxManEntry(cgrid_edge, part, nvars-1, 
                                           hypre_BoxArraySize(cboxes)-1, myproc, &entry);
   hypre_SStructBoxManEntryGetGlobalCSRank(entry, hypre_BoxIMax(cbox), &jupper);

   HYPRE_IJMatrixCreate(comm, ilower, iupper, jlower, jupper, &edge_Edge);
   HYPRE_IJMatrixSetObjectType(edge_Edge , HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(edge_Edge);

   /* Face_iedge. Only needed in 3-d. */
   if (ndim == 3)
   {
      /* lower rank */
      part= 0;
      box = 0;
      hypre_SStructGridBoxProcFindBoxManEntry(cgrid_face, part, 0, box, myproc, &entry);

      p_cgrid  = hypre_SStructGridPGrid(cgrid_face, part);
      var_cgrid= hypre_SStructPGridSGrid(p_cgrid, 0);
      cboxes   = hypre_StructGridBoxes(var_cgrid);
      cbox     = hypre_BoxArrayBox(cboxes, 0);
      hypre_SStructBoxManEntryGetGlobalCSRank(entry, hypre_BoxIMin(cbox), &ilower);

      hypre_SStructGridBoxProcFindBoxManEntry(fgrid_edge, part, 0, box, myproc, &entry);
      p_fgrid = hypre_SStructGridPGrid(fgrid_edge, part);
      var_fgrid= hypre_SStructPGridSGrid(p_fgrid, 0);
      fboxes   = hypre_StructGridBoxes(var_fgrid);
      fbox     = hypre_BoxArrayBox(fboxes, 0);
      hypre_SStructBoxManEntryGetGlobalCSRank(entry, hypre_BoxIMin(fbox), &jlower);

      /* upper rank */
      part= nparts-1;
      p_cgrid = hypre_SStructGridPGrid(cgrid_face, part);
      nvars   = hypre_SStructPGridNVars(p_cgrid);
      var_cgrid= hypre_SStructPGridSGrid(p_cgrid, nvars-1);
      cboxes   = hypre_StructGridBoxes(var_cgrid);
      cbox    = hypre_BoxArrayBox(cboxes, hypre_BoxArraySize(cboxes)-1);

      hypre_SStructGridBoxProcFindBoxManEntry(cgrid_face, part, nvars-1, 
                                              hypre_BoxArraySize(cboxes)-1, myproc, &entry);
      hypre_SStructBoxManEntryGetGlobalCSRank(entry, hypre_BoxIMax(cbox), &iupper);
   
      p_fgrid = hypre_SStructGridPGrid(fgrid_edge, part);
      nvars   = hypre_SStructPGridNVars(p_fgrid);

      var_fgrid= hypre_SStructPGridSGrid(p_fgrid, nvars-1);
      fboxes   = hypre_StructGridBoxes(var_fgrid);
      fbox     = hypre_BoxArrayBox(fboxes, hypre_BoxArraySize(fboxes)-1);

      hypre_SStructGridBoxProcFindBoxManEntry(fgrid_edge, part, nvars-1, 
                                              hypre_BoxArraySize(fboxes)-1, myproc, &entry);
      hypre_SStructBoxManEntryGetGlobalCSRank(entry, hypre_BoxIMax(fbox), &jupper);

      HYPRE_IJMatrixCreate(comm, ilower, iupper, jlower, jupper, &Face_iedge);
      HYPRE_IJMatrixSetObjectType(Face_iedge, HYPRE_PARCSR);
      HYPRE_IJMatrixInitialize(Face_iedge);
   }

   /* Element_Face. Only for 3-d since Element_Edge= Element_Face in 2-d. */
   /* lower rank */
   if (ndim == 3)
   {
      part= 0;
      box = 0;
      hypre_SStructGridBoxProcFindBoxManEntry(cgrid_element, part, 0, box, 
                                              myproc, &entry);

      p_cgrid  = hypre_SStructGridPGrid(cgrid_element, part);
      var_cgrid= hypre_SStructPGridSGrid(p_cgrid, 0);
      cboxes   = hypre_StructGridBoxes(var_cgrid);
      cbox     = hypre_BoxArrayBox(cboxes, 0);
      hypre_SStructBoxManEntryGetGlobalCSRank(entry, hypre_BoxIMin(cbox), &ilower);

      hypre_SStructGridBoxProcFindBoxManEntry(cgrid_face, part, 0, box, 
                                              myproc, &entry);
      p_cgrid = hypre_SStructGridPGrid(cgrid_face, part);
      var_cgrid= hypre_SStructPGridSGrid(p_cgrid, 0);
      cboxes  = hypre_StructGridBoxes(var_cgrid);
      cbox    = hypre_BoxArrayBox(cboxes, 0);
      hypre_SStructBoxManEntryGetGlobalCSRank(entry, hypre_BoxIMin(cbox), &jlower);

      /* upper rank */
      part= nparts-1;
      p_cgrid = hypre_SStructGridPGrid(cgrid_element, part);
      nvars   = hypre_SStructPGridNVars(p_cgrid);
      var_cgrid= hypre_SStructPGridSGrid(p_cgrid, nvars-1);
      cboxes  = hypre_StructGridBoxes(var_cgrid);
      cbox    = hypre_BoxArrayBox(cboxes, hypre_BoxArraySize(cboxes)-1);

      hypre_SStructGridBoxProcFindBoxManEntry(cgrid_element, part, nvars-1, 
                                              hypre_BoxArraySize(cboxes)-1, myproc, &entry);
      hypre_SStructBoxManEntryGetGlobalCSRank(entry, hypre_BoxIMax(cbox), &iupper);

      p_cgrid = hypre_SStructGridPGrid(cgrid_face, part);
      nvars   = hypre_SStructPGridNVars(p_cgrid);
      var_cgrid= hypre_SStructPGridSGrid(p_cgrid, nvars-1);
      cboxes  = hypre_StructGridBoxes(var_cgrid);
      cbox    = hypre_BoxArrayBox(cboxes, hypre_BoxArraySize(cboxes)-1);

      hypre_SStructGridBoxProcFindBoxManEntry(cgrid_face, part, nvars-1, 
                                              hypre_BoxArraySize(cboxes)-1, myproc, &entry);
      hypre_SStructBoxManEntryGetGlobalCSRank(entry, hypre_BoxIMax(cbox), &jupper);

      HYPRE_IJMatrixCreate(comm, ilower, iupper, jlower, jupper, &Element_Face);
      HYPRE_IJMatrixSetObjectType(Element_Face, HYPRE_PARCSR);
      HYPRE_IJMatrixInitialize(Element_Face);
   }

   /* Element_Edge. Same for 2-d and 3-d. */
   /* lower rank */
   part= 0;
   box = 0;
   hypre_SStructGridBoxProcFindBoxManEntry(cgrid_element, part, 0, box, myproc, &entry);

   p_cgrid  = hypre_SStructGridPGrid(cgrid_element, part);
   var_cgrid= hypre_SStructPGridSGrid(p_cgrid, 0);
   cboxes   = hypre_StructGridBoxes(var_cgrid);
   cbox     = hypre_BoxArrayBox(cboxes, 0);
   hypre_SStructBoxManEntryGetGlobalCSRank(entry, hypre_BoxIMin(cbox), &ilower);

   hypre_SStructGridBoxProcFindBoxManEntry(cgrid_edge, part, 0, box, myproc, &entry);
   p_cgrid = hypre_SStructGridPGrid(cgrid_edge, part);
   var_cgrid= hypre_SStructPGridSGrid(p_cgrid, 0);
   cboxes  = hypre_StructGridBoxes(var_cgrid);
   cbox    = hypre_BoxArrayBox(cboxes, 0);
   hypre_SStructBoxManEntryGetGlobalCSRank(entry, hypre_BoxIMin(cbox), &jlower);

   /* upper rank */
   part= nparts-1;
   p_cgrid = hypre_SStructGridPGrid(cgrid_element, part);
   nvars   = hypre_SStructPGridNVars(p_cgrid);
   var_cgrid= hypre_SStructPGridSGrid(p_cgrid, nvars-1);
   cboxes  = hypre_StructGridBoxes(var_cgrid);
   cbox    = hypre_BoxArrayBox(cboxes, hypre_BoxArraySize(cboxes)-1);

   hypre_SStructGridBoxProcFindBoxManEntry(cgrid_element, part, nvars-1, 
                                           hypre_BoxArraySize(cboxes)-1, myproc, &entry);
   hypre_SStructBoxManEntryGetGlobalCSRank(entry, hypre_BoxIMax(cbox), &iupper);

   p_cgrid = hypre_SStructGridPGrid(cgrid_edge, part);
   nvars   = hypre_SStructPGridNVars(p_cgrid);
   var_cgrid= hypre_SStructPGridSGrid(p_cgrid, nvars-1);
   cboxes  = hypre_StructGridBoxes(var_cgrid);
   cbox    = hypre_BoxArrayBox(cboxes, hypre_BoxArraySize(cboxes)-1);

   hypre_SStructGridBoxProcFindBoxManEntry(cgrid_edge, part, nvars-1, 
                                           hypre_BoxArraySize(cboxes)-1, myproc, &entry);
   hypre_SStructBoxManEntryGetGlobalCSRank(entry, hypre_BoxIMax(cbox), &jupper);

   HYPRE_IJMatrixCreate(comm, ilower, iupper, jlower, jupper, &Element_Edge);
   HYPRE_IJMatrixSetObjectType(Element_Edge, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(Element_Edge);

   /*------------------------------------------------------------------------------
    * fill up the parcsr matrices.
    *------------------------------------------------------------------------------*/
   /* count the number of connections, i.e., the columns 
    * no. of interior edges per face, or no. of interior edges per cell.
    * Need to distinguish between 2 and 3-d. */
   if (ndim == 3)
   {
      n_xFace_iedges= (rfactor[1]-1)*rfactor[2] + (rfactor[2]-1)*rfactor[1];
      n_yFace_iedges= (rfactor[0]-1)*rfactor[2] + (rfactor[2]-1)*rfactor[0];
      n_zFace_iedges= (rfactor[1]-1)*rfactor[0] + (rfactor[0]-1)*rfactor[1];
      n_Cell_iedges = (rfactor[2]-1)*n_zFace_iedges + 
         rfactor[2]*(rfactor[0]-1)*(rfactor[1]-1);
   
      nFaces_iedges = nxFaces*n_xFace_iedges + nyFaces*n_yFace_iedges + 
         nzFaces*n_zFace_iedges;
      nElements_iedges= nElements * n_Cell_iedges;
      nEdges_edges    = nxEdges*rfactor[0] + nyEdges*rfactor[1] + nzEdges*rfactor[2];
   }
   else
   {
      n_Cell_iedges = (rfactor[0]-1)*rfactor[1] + (rfactor[1]-1)*rfactor[0];
      nElements_iedges= nElements * n_Cell_iedges;

      /* edges= faces in 2-d are oriented differently; hence, the different
         formula to compute nEdges_edges. */
      nEdges_edges= nxEdges*rfactor[1] + nyEdges*rfactor[0]; 
   }
   
   if (ndim == 3)
   {
      iFace   = hypre_CTAlloc(HYPRE_Int, nFaces);
   }
   iEdge   = hypre_CTAlloc(HYPRE_Int, nEdges);
   iElement= hypre_CTAlloc(HYPRE_Int, nElements);

   /* array structures needed for forming ij_matrices */

   /* Element_edge. Same for 2-d and 3-d. */
   ncols_Elementedge= hypre_CTAlloc(HYPRE_Int, nElements);
   for (i= 0; i< nElements; i++)
   {
      ncols_Elementedge[i]= n_Cell_iedges;
   }
   jElement_edge    = hypre_CTAlloc(HYPRE_Int, nElements_iedges);
   vals_Elementedge = hypre_CTAlloc(double, nElements_iedges);

   /*---------------------------------------------------------------------------
    * Fill up the row/column ranks of Element_edge. Will need to distinguish
    * between 2-d and 3-d.
    *      Loop over the coarse element grid
    *        a) Refine the coarse cell and grab the fine cells that will contain
    *           the fine edges.
    *           To obtain the correct coarse-to-fine cell index mapping, we
    *           map [loopi,loopj,loopk] to the fine cell grid and then adjust
    *           so that the final mapped fine cell is the one on the upper
    *           corner of the agglomerate. Will need to determine the fine box
    *           corresponding to the coarse box.
    *        b) loop map these fine cells and find the ranks of the fine edges.
    *---------------------------------------------------------------------------*/
   nElements= 0;
   nElements_iedges= 0;
   for (part= 0; part< nparts; part++)
   {
      if (ndim == 3) 
      {
         p_cgrid      = hypre_SStructGridPGrid(cgrid_edge, part);  /* Edge grid */
         Edge_nvars   = hypre_SStructPGridNVars(p_cgrid);
         Edge_vartypes= hypre_SStructPGridVarTypes(p_cgrid);
      }
      else if (ndim == 2) /* edge is a face in 2-d*/
      {
         p_cgrid      = hypre_SStructGridPGrid(cgrid_face, part);  /* Face grid */
         Face_nvars   = hypre_SStructPGridNVars(p_cgrid);
         Face_vartypes= hypre_SStructPGridVarTypes(p_cgrid);
      }

      p_cgrid   = hypre_SStructGridPGrid(cgrid_element, part);  /* ccell grid */
      var_cgrid = hypre_SStructPGridCellSGrid(p_cgrid);
      cboxes    = hypre_StructGridBoxes(var_cgrid);

      p_fgrid   = hypre_SStructGridPGrid(fgrid_element, part);  /* fcell grid */
      var_fgrid = hypre_SStructPGridCellSGrid(p_fgrid);
      fboxes    = hypre_StructGridBoxes(var_fgrid);

      hypre_ForBoxI(i, cboxes)
      {
         cbox= hypre_BoxArrayBox(cboxes, i);
         hypre_BoxGetSize(cbox, loop_size);
         hypre_CopyIndex(hypre_BoxIMin(cbox), cstart);

         /* determine which fine box cbox has coarsened from. Obtained from
            cfbox_mapping. */ 
         fboxi= cfbox_mapping[part][i];
         fbox = hypre_BoxArrayBox(fboxes, fboxi);

         /**********************************************************************
          * determine the shift to get the correct c-to-f cell index map:
          *    d= hypre_BoxIMin(fbox)[j]%rfactor[j]*sign(hypre_BoxIMin(fbox)[j])
          *    stride[j]= d-1  if d>0
          *    stride[j]= rfactor[j]-1+d  if d<=0.
          * This is upper_shifts[part][fboxi].
          **********************************************************************/
         hypre_ClearIndex(stride);
         hypre_CopyIndex(upper_shifts[part][fboxi], stride);

         /* loop over each cell and find the row rank of Element_edge and then
            the column ranks of the connected fine edges. */
         hypre_BoxLoop0Begin(loop_size);
#if 0
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,cindex,findex,entry,rank,nElements,low_index,t,hi_index,var,m,k,j,var_index,nElements_iedges
#include "hypre_box_smp_forloop.h"
#else
         hypre_BoxLoopSetOneBlock();
#endif
         hypre_BoxLoop0For(loopi, loopj, loopk)
         {
            hypre_SetIndex(cindex, loopi, loopj, loopk);
            hypre_AddIndex(cindex, cstart, cindex);

            /* refined cindex to get the correct upper fine index */
            hypre_StructMapCoarseToFine(cindex, zero_index, rfactor, findex);
            hypre_AddIndex(findex, stride, findex);
             
            /* Element(i,j,k) rank */
            hypre_SStructGridFindBoxManEntry(cgrid_element, part, cindex, 0, &entry);
            hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank, matrix_type);
            iElement[nElements]= rank;
            nElements++;

            /* Element_iedge columns: 3-d, x_edges, y_edges, and z_edges. */
            if (ndim == 3)
            {
               hypre_SetIndex(low_index, findex[0]-rfactor[0]+1,
                              findex[1]-rfactor[1]+1,
                              findex[2]-rfactor[2]+1);

               for (t= 0; t< Edge_nvars; t++)
               {
                  hypre_CopyIndex(findex, hi_index);
                  var= Edge_vartypes[t]; /* c & f edges enumerated the same */

                  /* determine looping extents over the refined cells that 
                     will have fine edges. */
                  switch (var)
                  {
                     case 5:  /* x_edges */
                     {
                        hi_index[1]-= 1;
                        hi_index[2]-= 1;
                        break;
                     }
                     case 6:  /* y_edges */
                     {
                        hi_index[0]-= 1;
                        hi_index[2]-= 1;
                        break;
                     }
                     case 7:  /* z_edges */
                     {
                        hi_index[0]-= 1;
                        hi_index[1]-= 1;
                        break;
                     }
                  }   /* switch (var) */

                  /* column ranks. */
                  for (m= low_index[2]; m<= hi_index[2]; m++)
                  {
                     for (k= low_index[1]; k<= hi_index[1]; k++)
                     {
                        for (j= low_index[0]; j<= hi_index[0]; j++)
                        {
                           hypre_SetIndex(var_index, j, k, m);
                           hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index, 
                                                            t, &entry);
                           hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, 
                                                                 &rank, matrix_type);
                           jElement_edge[nElements_iedges]= rank;
                           nElements_iedges++;
                        }  /* for (j= findex[0]; j<= hi_index[0]; j++) */
                     }     /* for (k= findex[1]; k<= hi_index[1]; k++) */
                  }        /* for (m= findex[2]; m<= hi_index[2]; m++) */
               }           /* for (t= 0; t< Edge_nvars; t++) */
            }              /* if (ndim == 3) */
 
            else if (ndim == 2) /* only x & y faces */
            {
               hypre_SetIndex(low_index, findex[0]-rfactor[0]+1,
                              findex[1]-rfactor[1]+1,
                              findex[2]);

               for (t= 0; t< Face_nvars; t++)
               {
                  hypre_CopyIndex(findex, hi_index);
                  var= Face_vartypes[t]; /* c & f faces enumerated the same */

                  switch (var) /* note: hi_index computed differently in 2-d */
                  {
                     case 2:  /* x_faces */
                     {
                        hi_index[0]-= 1;
                        break;
                     }
                     case 3:  /* y_edges */
                     {
                        hi_index[1]-= 1;
                        break;
                     }
                  }   /* switch (var) */

                  /* column ranks. */
                  for (k= low_index[1]; k<= hi_index[1]; k++)
                  {
                     for (j= low_index[0]; j<= hi_index[0]; j++)
                     {
                        hypre_SetIndex(var_index, j, k, findex[2]);
                        hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index, 
                                                         t, &entry);
                        hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, 
                                                              &rank, matrix_type);
                        jElement_edge[nElements_iedges]= rank;
                        nElements_iedges++;
                     }  /* for (j= findex[0]; j<= hi_index[0]; j++) */
                  }     /* for (k= findex[1]; k<= hi_index[1]; k++) */
               }        /* for (t= 0; t< Face_nvars; t++) */
            }           /* if (ndim == 2) */
         }
         hypre_BoxLoop0End();
      }  /* hypre_ForBoxI(i, cboxes) */
   }     /* for (part= 0; part< nparts; part++) */

   HYPRE_IJMatrixSetValues(Element_iedge, nElements, ncols_Elementedge,
                           (const HYPRE_Int*) iElement, (const HYPRE_Int*) jElement_edge,
                           (const double*) vals_Elementedge);
   HYPRE_IJMatrixAssemble((HYPRE_IJMatrix) Element_iedge);

   hypre_TFree(ncols_Elementedge);
   hypre_TFree(jElement_edge);
   hypre_TFree(vals_Elementedge);

   /* Face_edge */
   /*------------------------------------------------------------------------------
    * Fill out Face_edge a row at a time. Since we have different Face types
    * so that the size of the cols change depending on what type the Face
    * is, we need to loop over the grids and take a count of the col elements.
    * Loop over the coarse face grids and add up the number of interior edges.
    * Will compute only for 3-d. In 2-d, these structures are obtained for
    * Edge_edge.
    *------------------------------------------------------------------------------*/
   if (ndim == 3)
   {
      ncols_Faceedge = hypre_CTAlloc(HYPRE_Int, nFaces);
      nFaces= 0;
      j= 0; 
      for (part= 0; part< nparts; part++)
      {
         p_cgrid      = hypre_SStructGridPGrid(cgrid_face, part);  /* Face grid */
         Face_nvars   = hypre_SStructPGridNVars(p_cgrid);
         Face_vartypes= hypre_SStructPGridVarTypes(p_cgrid);

         p_fgrid   = hypre_SStructGridPGrid(fgrid_edge, part);  
         var_fgrid = hypre_SStructPGridCellSGrid(p_fgrid);
         fboxes    = hypre_StructGridBoxes(var_fgrid);

         for (t= 0; t< Face_nvars; t++)
         {
            var= Face_vartypes[t];
            var_cgrid= hypre_SStructPGridSGrid(p_cgrid, t);
            k= hypre_StructGridLocalSize(var_cgrid);

            switch(var)
            {
               case 2: /* x_Faces (i,j,k) then (i-1,j,k), contain y,z edges */
               {
                  for (i= 0; i< k; i++)
                  {
                     /* y_iedge connections to x_Face */
                     ncols_Faceedge[nFaces]= (rfactor[2]-1)*rfactor[1];

                     /* z_iedge connections to x_Face */
                     ncols_Faceedge[nFaces]+= rfactor[2]*(rfactor[1]-1);

                     j+= ncols_Faceedge[nFaces];
                     nFaces++;
                  }
                  break;
               }   /* case 2 */

               case 3: /* y_Faces (i,j,k) then (i,j-1,k), contain x,z edges */
               {
                  for (i= 0; i< k; i++)
                  {
                     /* x_iedge connections to y_Face */
                     ncols_Faceedge[nFaces]= (rfactor[2]-1)*rfactor[0];

                     /* z_iedge connections to y_Face */
                     ncols_Faceedge[nFaces]+= rfactor[2]*(rfactor[0]-1);

                     j+= ncols_Faceedge[nFaces];
                     nFaces++;
                  }
                  break;
               }   /* case 3 */

               case 4: /* z_Faces (i,j,k) then (i,j,k-1), contain x,y edges */
               {
                  for (i= 0; i< k; i++)
                  {
                     /* x_iedge connections to z_Face */
                     ncols_Faceedge[nFaces]= (rfactor[1]-1)*rfactor[0];

                     /* y_iedge connections to z_Face */
                     ncols_Faceedge[nFaces]+= rfactor[1]*(rfactor[0]-1);

                     j+= ncols_Faceedge[nFaces];
                     nFaces++;
                  }
                  break;
               }   /* case 4 */

            } /* switch(var) */
         }    /* for (t= 0; t< Face_nvars; t++) */
      }       /* for (part= 0; part< nparts; part++) */

      jFace_edge= hypre_CTAlloc(HYPRE_Int, j);
      vals_Faceedge= hypre_CTAlloc(double, j);
      for (i= 0; i< j; i++)
      {
         vals_Faceedge[i]= 1.0;
      }

      /*---------------------------------------------------------------------------
       * Fill up the row/column ranks of Face_edge. 
       *      Loop over the coarse Cell grid
       *        a) for each Cell box, stretch to a Face box
       *        b) for each coarse face, if it is on the proc, map it to a
       *           coarse cell (add the variable offset).
       *        c) refine the coarse cell and grab the fine cells that will contain
       *           the fine edges. Refining requires a shifting dependent on the
       *           begining index of the fine box.
       *        d) map these fine cells to the fine edges.
       *---------------------------------------------------------------------------*/
      nFaces       = 0;
      nFaces_iedges= 0;
      for (part= 0; part< nparts; part++)
      {
         p_cgrid      = hypre_SStructGridPGrid(cgrid_face, part);  /* Face grid */
         Face_nvars   = hypre_SStructPGridNVars(p_cgrid);
         Face_vartypes= hypre_SStructPGridVarTypes(p_cgrid);

         for (t= 0; t< Face_nvars; t++)
         {
            var= Face_vartypes[t];
            var_cgrid = hypre_SStructPGridCellSGrid(p_cgrid);
            cboxes   = hypre_StructGridBoxes(var_cgrid);

            /* to eliminate comparisons, take the switch outside of the loop. */
            switch(var)
            {
               case 2:  /* x_Faces-> y_iedges, z_iedges */
               {
                  hypre_ForBoxI(i, cboxes)
                  {
                     cbox= hypre_BoxArrayBox(cboxes, i);
                     hypre_CopyBox(cbox, &copy_box);
                     hypre_SubtractIndex(hypre_BoxIMin(&copy_box), varoffsets[var],
                                         hypre_BoxIMin(&copy_box));

                     hypre_BoxGetSize(&copy_box, loop_size);
                     hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                     /* determine which fine box cbox has coarsened from */
                     fboxi= cfbox_mapping[part][i];
                     fbox = hypre_BoxArrayBox(fboxes, fboxi);

                     /**********************************************************
                      * determine the shift to get the correct c-to-f cell 
                      * index map. This is upper_shifts[part][fboxi].
                      **********************************************************/
                     hypre_ClearIndex(stride);
                     hypre_CopyIndex(upper_shifts[part][fboxi], stride);

                     hypre_BoxLoop0Begin(loop_size);
#if 0
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,cindex,entry,rank,nFaces,cell_index,findex,j,ilower,k,var_index,nFaces_iedges
#include "hypre_box_smp_forloop.h"
#else
                     hypre_BoxLoopSetOneBlock();
#endif
                     hypre_BoxLoop0For(loopi, loopj, loopk)
                     {
                        hypre_SetIndex(cindex, loopi, loopj, loopk);
                        hypre_AddIndex(cindex, start, cindex);

                        hypre_SStructGridFindBoxManEntry(cgrid_face, part, cindex, t, &entry);
                        hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank, matrix_type);

                        /* check if rank on proc before continuing */
                        if ((rank <= cupper_ranks[part][var]) &&
                            (rank >= clower_ranks[part][var]))
                        {
                           iFace[nFaces]= rank;
                           nFaces++;

                           /* transform face index to cell index */
                           hypre_AddIndex(cindex, varoffsets[var], cell_index);

                           /* Refine the coarse cell to the upper fine index. The face will
                              be on the "lower end" fine cells, i.e., the slab box starting
                              with fine index cell_index. The fine edges will be on the
                              lower x of the fine cell, e.g., with fine cell (i,j,k),
                              y_iedge (i-1,j,k) & z_iedge (i-1,j,k). */
                           hypre_StructMapCoarseToFine(cell_index, zero_index,
                                                       rfactor, findex);
                           hypre_AddIndex(findex, stride, findex);

                           /* cell_index was refined to the upper fine index. Shift
                              back to the lower end, subtract (rfactor-1). */
                           for (j= 0; j< ndim; j++)
                           {
                              findex[j]-= rfactor[j]-1;
                           }

                           /* y_iedges */
                           ilower= findex[0]-1;
                           for (k= 0; k< rfactor[2]-1; k++)
                           {
                              for (j= 0; j< rfactor[1]; j++)
                              {
                                 hypre_SetIndex(var_index, ilower, j+findex[1], k+findex[2]);
                                 hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                                  vartype_map[6], &entry);
                                 hypre_SStructBoxManEntryGetGlobalRank(entry, var_index,
                                                                       &rank, matrix_type);
                                 jFace_edge[nFaces_iedges]= rank;
                                 nFaces_iedges++;
                              }
                           }

                           /* z_iedges */
                           for (k= 0; k< rfactor[2]; k++)
                           {
                              for (j= 0; j< rfactor[1]-1; j++)
                              {
                                 hypre_SetIndex(var_index, ilower, j+findex[1], k+findex[2]);
                                 hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                                  vartype_map[7], &entry);
                                 hypre_SStructBoxManEntryGetGlobalRank(entry, var_index,
                                                                       &rank, matrix_type);
                                 jFace_edge[nFaces_iedges]= rank;
                                 nFaces_iedges++;
                              }
                           }
                        }  /* if ((rank <= cupper_ranks[part][var]) &&
                              (rank >= clower_ranks[part][var])) */
                     }

                     hypre_BoxLoop0End();
                  }  /* hypre_ForBoxI(i, cboxes) */
                  break;
               }   /* case 2:  x_Faces-> y_iedges, z_iedges */

               case 3:  /* y_Faces-> x_iedges, z_iedges */
               {
                  hypre_ForBoxI(i, cboxes)
                  {
                     cbox= hypre_BoxArrayBox(cboxes, i);
                     hypre_CopyBox(cbox, &copy_box);
                     hypre_SubtractIndex(hypre_BoxIMin(&copy_box), varoffsets[var],
                                         hypre_BoxIMin(&copy_box));

                     hypre_BoxGetSize(&copy_box, loop_size);
                     hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                     /* determine which fine box cbox has coarsened from */
                     fboxi= cfbox_mapping[part][i];
                     fbox = hypre_BoxArrayBox(fboxes, fboxi);

                     /**********************************************************
                      * determine the shift to get the correct c-to-f cell 
                      * index map. This is upper_shifts[part][fboxi].
                      **********************************************************/
                     hypre_ClearIndex(stride);
                     hypre_CopyIndex(upper_shifts[part][fboxi], stride);

                     hypre_BoxLoop0Begin(loop_size);
#if 0
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,cindex,entry,rank,nFaces,cell_index,findex,j,ilower,k,var_index,nFaces_iedges
#include "hypre_box_smp_forloop.h"
#else
                     hypre_BoxLoopSetOneBlock();
#endif
                     hypre_BoxLoop0For(loopi, loopj, loopk)
                     {
                        hypre_SetIndex(cindex, loopi, loopj, loopk);
                        hypre_AddIndex(cindex, start, cindex);

                        hypre_SStructGridFindBoxManEntry(cgrid_face, part, cindex, t, &entry);
                        hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank, matrix_type);
                        /* check if rank on proc before continuing */
                        if ((rank <= cupper_ranks[part][var]) &&
                            (rank >= clower_ranks[part][var]))
                        {
                           iFace[nFaces]= rank;
                           nFaces++;

                           /* transform face index to cell index */
                           hypre_AddIndex(cindex, varoffsets[var], cell_index);

                           /* Refine the coarse cell to the upper fine index. The face will
                              be on the "lower end" fine cells, i.e., the slab box starting
                              with fine index cell_index. The fine edges will be on the
                              lower x of the fine cell, e.g., with fine cell (i,j,k),
                              y_iedge (i-1,j,k) & z_iedge (i-1,j,k). */
                           hypre_StructMapCoarseToFine(cell_index, zero_index,
                                                       rfactor, findex);
                           hypre_AddIndex(findex, stride, findex);

                           /* cell_index is refined to the upper fine index. Shift
                              back to the lower end, subtract (rfactor-1). */
                           for (j= 0; j< ndim; j++)
                           {
                              findex[j]-= rfactor[j]-1;
                           }

                           /* x_iedges */
                           ilower= findex[1]-1;
                           for (k= 0; k< rfactor[2]-1; k++)
                           {
                              for (j= 0; j< rfactor[0]; j++)
                              {
                                 hypre_SetIndex(var_index, j+findex[0], ilower, k+findex[2]);
                                 hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                                  vartype_map[5], &entry);
                                 hypre_SStructBoxManEntryGetGlobalRank(entry, var_index,
                                                                       &rank, matrix_type);
                                 jFace_edge[nFaces_iedges]= rank;
                                 nFaces_iedges++;
                              }
                           }

                           /* z_iedges */
                           for (k= 0; k< rfactor[2]; k++)
                           {
                              for (j= 0; j< rfactor[0]-1; j++)
                              {
                                 hypre_SetIndex(var_index, j+findex[0], ilower, k+findex[2]);
                                 hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                                  vartype_map[7], &entry);
                                 hypre_SStructBoxManEntryGetGlobalRank(entry, var_index,
                                                                       &rank, matrix_type);
                                 jFace_edge[nFaces_iedges]= rank;
                                 nFaces_iedges++;
                              }
                           }
                        }  /* if ((rank <= cupper_ranks[part][var]) &&
                              (rank >= clower_ranks[part][var])) */
                     }

                     hypre_BoxLoop0End();
                  }  /* hypre_ForBoxI(i, cboxes) */
                  break;
               }   /* case 3:  y_Faces-> x_iedges, z_iedges */

               case 4:  /* z_Faces-> x_iedges, y_iedges */
               {
                  hypre_ForBoxI(i, cboxes)
                  {
                     cbox= hypre_BoxArrayBox(cboxes, i);
                     hypre_CopyBox(cbox, &copy_box);
                     hypre_SubtractIndex(hypre_BoxIMin(&copy_box), varoffsets[var],
                                         hypre_BoxIMin(&copy_box));

                     hypre_BoxGetSize(&copy_box, loop_size);
                     hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                     /* determine which fine box cbox has coarsened from */
                     fboxi= cfbox_mapping[part][i];
                     fbox = hypre_BoxArrayBox(fboxes, fboxi);

                     /**********************************************************
                      * determine the shift to get the correct c-to-f cell
                      * index map. This is upper_shifts[part][fboxi].
                      **********************************************************/
                     hypre_ClearIndex(stride);
                     hypre_CopyIndex(upper_shifts[part][fboxi], stride);

                     hypre_BoxLoop0Begin(loop_size);
#if 0
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,entry,rank,nFaces,cell_index,findex,j,ilower,k,var_index,nFaces_iedges
#include "hypre_box_smp_forloop.h"
#else
                     hypre_BoxLoopSetOneBlock();
#endif
                     hypre_BoxLoop0For(loopi, loopj, loopk)
                     {
                        hypre_SetIndex(cindex, loopi, loopj, loopk);
                        hypre_AddIndex(cindex, start, cindex);

                        hypre_SStructGridFindBoxManEntry(cgrid_face, part, cindex, t, &entry);
                        hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank, matrix_type);

                        /* check if rank on proc before continuing */
                        if ((rank <= cupper_ranks[part][var]) &&
                            (rank >= clower_ranks[part][var]))
                        {
                           iFace[nFaces]= rank;
                           nFaces++;

                           /* transform face index to cell index */
                           hypre_AddIndex(cindex, varoffsets[var], cell_index);

                           /* Refine the coarse cell to the upper fine index. The face will
                              be on the "lower end" fine cells, i.e., the slab box starting
                              with fine index cell_index. The fine edges will be on the
                              lower x of the fine cell, e.g., with fine cell (i,j,k),
                              y_iedge (i-1,j,k) & z_iedge (i-1,j,k). */
                           hypre_StructMapCoarseToFine(cell_index, zero_index,
                                                       rfactor, findex);
                           hypre_AddIndex(findex, stride, findex);

                           /* cell_index is refined to the upper fine index. Shift
                              back to the lower end, subtract (rfactor-1). */
                           for (j= 0; j< ndim; j++)
                           {
                              findex[j]-= rfactor[j]-1;
                           }

                           /* x_iedges */
                           ilower= findex[2]-1;
                           for (k= 0; k< rfactor[1]-1; k++)
                           {
                              for (j= 0; j< rfactor[0]; j++)
                              {
                                 hypre_SetIndex(var_index, j+findex[0], k+findex[1], ilower);
                                 hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                                  vartype_map[5], &entry);
                                 hypre_SStructBoxManEntryGetGlobalRank(entry, var_index,
                                                                       &rank, matrix_type);
                                 jFace_edge[nFaces_iedges]= rank;
                                 nFaces_iedges++;
                              }
                           }

                           /* y_iedges */
                           for (k= 0; k< rfactor[1]; k++)
                           {
                              for (j= 0; j< rfactor[0]-1; j++)
                              {
                                 hypre_SetIndex(var_index, j+findex[0], k+findex[1], ilower);
                                 hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                                  vartype_map[6], &entry);
                                 hypre_SStructBoxManEntryGetGlobalRank(entry, var_index,
                                                                       &rank, matrix_type);
                                 jFace_edge[nFaces_iedges]= rank;
                                 nFaces_iedges++;
                              }
                           }
                        }  /* if ((rank <= cupper_ranks[part][var]) &&
                              (rank >= clower_ranks[part][var])) */
                     }
                     hypre_BoxLoop0End();
                  }  /* hypre_ForBoxI(i, cboxes) */
                  break;
               }   /* case 4:  z_Faces-> x_iedges, y_iedges */

            }   /* switch(var) */
         }      /* for (t= 0; t< Face_nvars; t++) */
      }         /* for (part= 0; part< nparts; part++) */

      HYPRE_IJMatrixSetValues(Face_iedge, nFaces, ncols_Faceedge,
                              (const HYPRE_Int*) iFace, (const HYPRE_Int*) jFace_edge,
                              (const double*) vals_Faceedge);
      HYPRE_IJMatrixAssemble((HYPRE_IJMatrix) Face_iedge);

      hypre_TFree(ncols_Faceedge);
      hypre_TFree(iFace);
      hypre_TFree(jFace_edge);
      hypre_TFree(vals_Faceedge);
   }  /* if (ndim == 3) */

   /* Edge_edge */
   /*------------------------------------------------------------------------------
    * Count the Edge_edge connections. Will need to distinguish 2-d and 3-d.
    *------------------------------------------------------------------------------*/
   /* nEdges should be correct for 2-d & 3-d */
   ncols_Edgeiedge= hypre_CTAlloc(HYPRE_Int, nEdges);

   nEdges= 0; 
   k= 0;
   for (part= 0; part< nparts; part++)
   {
      /* Edge grid. In 2-d this will be the face grid, which is assumed to be
         in cgrid_edge. */
      p_cgrid      = hypre_SStructGridPGrid(cgrid_edge, part); 
      Edge_vartypes= hypre_SStructPGridVarTypes(p_cgrid);
      Edge_nvars   = hypre_SStructPGridNVars(p_cgrid);

      for (t= 0; t< Edge_nvars; t++)
      {
         var= Edge_vartypes[t];
         var_cgrid= hypre_SStructPGridSGrid(p_cgrid, t);
         j= hypre_StructGridLocalSize(var_cgrid);

         switch(var)
         {
            case 2:    /* 2-d, x_Face */
            {
               m= rfactor[1];
               break;
            }

            case 3:    /* 2-d, y_Face */
            {
               m= rfactor[0];
               break;
            }

            case 5:    /* 3-d, x_Edge */
            {
               m= rfactor[0];
               break;
            }

            case 6:    /* 3-d, y_Edge */
            {
               m= rfactor[1];
               break;
            }

            case 7:    /* 3-d, z_Edge */
            {
               m= rfactor[2];
               break;
            }
         }

         for (i= nEdges; i< nEdges+j; i++) /*fill in the column size for Edge */
         {
            ncols_Edgeiedge[i]= m;
            k+= m;
         }
         nEdges+= j;

      }  /* for (t= 0; t< Edge_nvars; t++) */
   }     /* for (part= 0; part< nparts; part++) */

   jEdge_iedge= hypre_CTAlloc(HYPRE_Int, k);
   vals_Edgeiedge= hypre_CTAlloc(double, k);
   for (i= 0; i< k; i++)
   {
      vals_Edgeiedge[i]= 1.0;
   }

   /*---------------------------------------------------------------------------
    * Fill up the row/column ranks of Edge_edge. Since a refinement of the
    * coarse edge index does not get the correct fine edge index, we need to 
    * map it to the cell grid. Recall, all variable grids are gotten by coarsening
    * a cell centred grid.
    *      Loop over the coarse Cell grid
    *        a) for each Cell box, map to an Edge box
    *        b) for each coarse Edge on my proc , map it to a coarse cell
    *           (add the variable offset).
    *        c) refine the coarse cell and grab the fine cells that will contain
    *           the fine edges.
    *        d) map these fine cells to the fine edges.
    *---------------------------------------------------------------------------*/

   nEdges       = 0;
   nEdges_iedges= 0;
   for (part= 0; part< nparts; part++)
   {
      p_cgrid      = hypre_SStructGridPGrid(cgrid_edge, part); 
      Edge_vartypes= hypre_SStructPGridVarTypes(p_cgrid);
      Edge_nvars   = hypre_SStructPGridNVars(p_cgrid);
      
      p_fgrid   = hypre_SStructGridPGrid(fgrid_edge, part);  
      var_fgrid = hypre_SStructPGridCellSGrid(p_fgrid);
      fboxes    = hypre_StructGridBoxes(var_fgrid);

      for (t= 0; t< Edge_nvars; t++)
      {
         var= Edge_vartypes[t];
         var_cgrid= hypre_SStructPGridCellSGrid(p_cgrid);
         cboxes   = hypre_StructGridBoxes(var_cgrid);

         hypre_ForBoxI(i, cboxes)
         {
            cbox= hypre_BoxArrayBox(cboxes, i);

            /*-------------------------------------------------------------------
             * extract the variable box by offsetting with var_offset. Note that
             * this may lead to a bigger variable domain than is on this proc.
             * Off-proc Edges will be checked to eliminate this problem.
             *-------------------------------------------------------------------*/
            hypre_CopyBox(cbox, &copy_box);
            hypre_SubtractIndex(hypre_BoxIMin(&copy_box), varoffsets[var],
                                hypre_BoxIMin(&copy_box));
            hypre_BoxGetSize(&copy_box, loop_size);
            hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

            /* determine which fine box cbox has coarsened from */
            fboxi= cfbox_mapping[part][i];
            fbox = hypre_BoxArrayBox(fboxes, fboxi);

            /**********************************************************
             * determine the shift to get the correct c-to-f cell
             * index map. This is upper_shifts[part][fboxi].
             **********************************************************/
            hypre_ClearIndex(stride);
            hypre_CopyIndex(upper_shifts[part][fboxi], stride);

            hypre_BoxLoop0Begin(loop_size);
#if 0
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,entry,rank,nEdges,cell_index,findex,j,var_index,m,entry,rank,nEdges_iedges
#include "hypre_box_smp_forloop.h"
#else
            hypre_BoxLoopSetOneBlock();
#endif
            hypre_BoxLoop0For(loopi, loopj, loopk)
            {
               hypre_SetIndex(cindex, loopi, loopj, loopk);
               hypre_AddIndex(cindex, start, cindex);

               /* row rank */
               hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex, t,
                                                &entry);
               hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                     matrix_type);

               /* check if rank on proc before continuing */
               if ((rank <= cupper_ranks[part][var]) &&
                   (rank >= clower_ranks[part][var]))
               {
                  iEdge[nEdges]= rank;
                  nEdges++;

                  hypre_AddIndex(cindex, varoffsets[var], cell_index);

                  /* refine cindex and then map back to variable index */
                  hypre_StructMapCoarseToFine(cell_index, zero_index, rfactor,
                                              findex);
                  hypre_AddIndex(findex, stride, findex);

                  /* cell_index is refined to the upper fine index. Shift
                     back to the lower end, subtract (rfactor-1). */
                  for (j= 0; j< ndim; j++)
                  {
                     findex[j]-= rfactor[j]-1;
                  }

                  hypre_SubtractIndex(findex, varoffsets[var], var_index);
                
                  switch(var)
                  {
                     case 2:    /* 2-d, x_face */
                     {
                        for (m= 0; m< rfactor[1]; m++)
                        {
                           hypre_SStructGridFindBoxManEntry(fgrid_edge, part,
                                                            var_index, t, &entry);
                           hypre_SStructBoxManEntryGetGlobalRank(entry, var_index,
                                                                 &rank, matrix_type);
                           jEdge_iedge[nEdges_iedges]= rank;
                           nEdges_iedges++;

                           /* increment the x component to get the next one in the
                              refinement cell. */
                           var_index[1]++;
                        }
                        break;
                     }

                     case 3:    /* 2-d, y_face */
                     {
                        for (m= 0; m< rfactor[0]; m++)
                        {
                           hypre_SStructGridFindBoxManEntry(fgrid_edge, part,
                                                            var_index, t, &entry);
                           hypre_SStructBoxManEntryGetGlobalRank(entry, var_index,
                                                                 &rank, matrix_type);
                           jEdge_iedge[nEdges_iedges]= rank;
                           nEdges_iedges++;

                           /* increment the y component to get the next one in the
                              refinement cell. */
                           var_index[0]++;
                        }
                        break;
                     }

                     case 5:    /* 3-d, x_edge */
                     {
                        for (m= 0; m< rfactor[0]; m++)
                        {
                           hypre_SStructGridFindBoxManEntry(fgrid_edge, part,
                                                            var_index, t, &entry);
                           hypre_SStructBoxManEntryGetGlobalRank(entry, var_index,
                                                                 &rank, matrix_type);
                           jEdge_iedge[nEdges_iedges]= rank;
                           nEdges_iedges++;

                           /* increment the x component to get the next one in the
                              refinement cell. */
                           var_index[0]++;
                        }
                        break;
                     }

                     case 6:    /* 3-d, y_edge */
                     {
                        for (m= 0; m< rfactor[1]; m++)
                        {
                           hypre_SStructGridFindBoxManEntry(fgrid_edge, part,
                                                            var_index, t, &entry);
                           hypre_SStructBoxManEntryGetGlobalRank(entry, var_index,
                                                                 &rank, matrix_type);
                           jEdge_iedge[nEdges_iedges]= rank;
                           nEdges_iedges++;

                           /* increment the y component to get the next one in the
                              refinement cell. */
                           var_index[1]++;
                        }
                        break;
                     }

                     case 7:    /* 3-d, z_edge */
                     {
                        for (m= 0; m< rfactor[2]; m++)
                        {
                           hypre_SStructGridFindBoxManEntry(fgrid_edge, part,
                                                            var_index, t, &entry);
                           hypre_SStructBoxManEntryGetGlobalRank(entry, var_index,
                                                                 &rank, matrix_type);
                           jEdge_iedge[nEdges_iedges]= rank;
                           nEdges_iedges++;

                           /* increment the z component to get the next one in the
                              refinement cell. */
                           var_index[2]++;
                        }
                        break;
                     }
                  }  /* switch(var) */

               }   /* if ((rank <= cupper_ranks[part][var]) &&
                      (rank >= clower_ranks[part][var])) */
            }
            hypre_BoxLoop0End();

         }  /* hypre_ForBoxI(i, cboxes) */
      }     /* for (t= 0; t< Edge_nvars; t++) */
   }        /* for (part= 0; part< nparts; part++) */
                     
   HYPRE_IJMatrixSetValues(Edge_iedge, nEdges, ncols_Edgeiedge,
                           (const HYPRE_Int*) iEdge, (const HYPRE_Int*) jEdge_iedge,
                           (const double*) vals_Edgeiedge);
   HYPRE_IJMatrixAssemble((HYPRE_IJMatrix) Edge_iedge);

   hypre_TFree(ncols_Edgeiedge);
   hypre_TFree(iEdge);
   hypre_TFree(jEdge_iedge);
   hypre_TFree(vals_Edgeiedge);

   /* Element_Face & Element_Edge. Element_Face only for 3-d. */
   if (ndim == 3)
   {
      ncols_ElementFace= hypre_CTAlloc(HYPRE_Int, nElements);
      j= 2*ndim;
      for (i= 0; i< nElements; i++)
      {
         ncols_ElementFace[i]= j;  /* 3-dim -> 6  */
      }

      j*= nElements;
      jElement_Face   = hypre_CTAlloc(HYPRE_Int, j);
      vals_ElementFace= hypre_CTAlloc(double, j);
      for (i= 0; i< j; i++)
      {
         vals_ElementFace[i]= 1.0;
      }
   }

   ncols_ElementEdge= hypre_CTAlloc(HYPRE_Int, nElements);
   j= 2*ndim;
   k= (ndim-1)*j;
   for (i= 0; i< nElements; i++)
   {
      ncols_ElementEdge[i]= k;  /* 2-dim -> 4; 3-dim -> 12 */
   }

   k*= nElements;
   jElement_Edge   = hypre_CTAlloc(HYPRE_Int, k);
   vals_ElementEdge= hypre_CTAlloc(double, k);
   for (i= 0; i< k; i++)
   {
      vals_ElementEdge[i]= 1.0;
   }

   /*---------------------------------------------------------------------------
    * Fill up the column ranks of ELement_Face and Element_Edge. Note that the
    * iElement has alrady been formed when filling Element_edge.
    *---------------------------------------------------------------------------*/
   nElements_Faces= 0;
   nElements_Edges= 0;
   for (part= 0; part< nparts; part++)
   {
      /* grab the nvars & vartypes for the face and edge variables */
      if (ndim == 3)
      {
         p_cgrid      = hypre_SStructGridPGrid(cgrid_face, part);  
         Face_nvars   = hypre_SStructPGridNVars(p_cgrid);
         Face_vartypes= hypre_SStructPGridVarTypes(p_cgrid);
      }

      p_cgrid      = hypre_SStructGridPGrid(cgrid_edge, part);  /* Edge grid */
      Edge_nvars   = hypre_SStructPGridNVars(p_cgrid);
      Edge_vartypes= hypre_SStructPGridVarTypes(p_cgrid);

      p_cgrid   = hypre_SStructGridPGrid(cgrid_element, part);  /* cell grid */
      var_cgrid = hypre_SStructPGridCellSGrid(p_cgrid);
      cboxes    = hypre_StructGridBoxes(var_cgrid);

      if (ndim == 3)
      {
         hypre_ForBoxI(i, cboxes)
         {
            cbox= hypre_BoxArrayBox(cboxes, i);
            hypre_BoxGetSize(cbox, loop_size);
            hypre_CopyIndex(hypre_BoxIMin(cbox), start);

            hypre_BoxLoop0Begin(loop_size);
#if 0
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,cindex,t,var,entry,rank,nElements_Faces,var_index
#include "hypre_box_smp_forloop.h"
#else
            hypre_BoxLoopSetOneBlock();
#endif
            hypre_BoxLoop0For(loopi, loopj, loopk)
            {
               hypre_SetIndex(cindex, loopi, loopj, loopk);
               hypre_AddIndex(cindex, start, cindex);

               /*-------------------------------------------------------------
                * jElement_Face: (i,j,k) then (i-1,j,k), (i,j-1,k), (i,j,k-1).
                *-------------------------------------------------------------*/
               for (t= 0; t< Face_nvars; t++)
               {
                  var= Face_vartypes[t];

                  hypre_SStructGridFindBoxManEntry(cgrid_face, part, cindex, t,
                                                   &entry);
                  hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                        matrix_type);
                  jElement_Face[nElements_Faces]= rank;
                  nElements_Faces++;

                  hypre_SubtractIndex(cindex, varoffsets[var], var_index);
                  hypre_SStructGridFindBoxManEntry(cgrid_face, part, var_index, t,
                                                   &entry);
                  hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                        matrix_type);
                  jElement_Face[nElements_Faces]= rank;
                  nElements_Faces++;
               }

            }
            hypre_BoxLoop0End();
         }  /* hypre_ForBoxI(i, cboxes) */
      }  /* if (ndim == 3) */

      /*-------------------------------------------------------------------
       * jElement_Edge: 
       *    3-dim
       *       x_Edge: (i,j,k) then (i,j-1,k), (i,j-1,k-1), (i,j,k-1)
       *       y_Edge: (i,j,k) then (i-1,j,k), (i-1,j,k-1), (i,j,k-1)
       *       z_Edge: (i,j,k) then (i,j-1,k), (i-1,j-1,k), (i-1,j,k)
       *
       *    2-dim
       *       x_Edge or x_Face: (i,j) then (i-1,j)
       *       y_Edge or y_Face: (i,j) then (i,j-1)
       *-------------------------------------------------------------------*/
      hypre_ForBoxI(i, cboxes)
      {
         cbox= hypre_BoxArrayBox(cboxes, i);
         hypre_BoxGetSize(cbox, loop_size);
         hypre_CopyIndex(hypre_BoxIMin(cbox), start);

         hypre_BoxLoop0Begin(loop_size);
#if 0
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,cindex,t,var,entry,rank,nElements_Edges,var_index
#include "hypre_box_smp_forloop.h"
#else
         hypre_BoxLoopSetOneBlock();
#endif
         hypre_BoxLoop0For(loopi, loopj, loopk)
         {
            hypre_SetIndex(cindex, loopi, loopj, loopk);
            hypre_AddIndex(cindex, start, cindex);

            for (t= 0; t< Edge_nvars; t++)
            {
               /* Edge (i,j,k) */
               var= Edge_vartypes[t];

               switch (var)
               {
                  case 2: /* x_Face= {(i,j), (i-1,j)} */
                  {
                     hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex, t,
                                                      &entry);
                     hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                           matrix_type);
                     jElement_Edge[nElements_Edges]= rank;
                     nElements_Edges++;

                     hypre_SubtractIndex(cindex, ishift, var_index);
                     hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index, t,
                                                      &entry);
                     hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                           matrix_type);
                     jElement_Edge[nElements_Edges]= rank;
                     nElements_Edges++;
                     break;
                  }

                  case 3: /* y_Face= {(i,j), (i,j-1)} */
                  {
                     hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex, t,
                                                      &entry);
                     hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                           matrix_type);
                     jElement_Edge[nElements_Edges]= rank;
                     nElements_Edges++;

                     hypre_SubtractIndex(cindex, jshift, var_index);
                     hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index, t,
                                                      &entry);
                     hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                           matrix_type);
                     jElement_Edge[nElements_Edges]= rank;
                     nElements_Edges++;
                     break;
                  }

                  case 5: /* "/" x_Edge={(i,j,k),(i,j-1,k),(i,j-1,k-1),(i,j,k-1)} */
                  {
                     hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex, t,
                                                      &entry);
                     hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                           matrix_type);
                     jElement_Edge[nElements_Edges]= rank;
                     nElements_Edges++;

                     hypre_SubtractIndex(cindex, jshift, var_index);
                     hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index, t,
                                                      &entry);
                     hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                           matrix_type);
                     jElement_Edge[nElements_Edges]= rank;
                     nElements_Edges++;

                     hypre_SubtractIndex(var_index, kshift, var_index);
                     hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index, t,
                                                      &entry);
                     hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                           matrix_type);
                     jElement_Edge[nElements_Edges]= rank;
                     nElements_Edges++;

                     hypre_AddIndex(var_index, jshift, var_index);
                     hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index, t,
                                                      &entry);
                     hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                           matrix_type);
                     jElement_Edge[nElements_Edges]= rank;
                     nElements_Edges++;
                     break;
                  }

                  case 6: /* "-" y_Edge={(i,j,k),(i-1,j,k),(i-1,j,k-1),(i,j,k-1)}*/
                  {
                     hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex, t,
                                                      &entry);
                     hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                           matrix_type);
                     jElement_Edge[nElements_Edges]= rank;
                     nElements_Edges++;

                     hypre_SubtractIndex(cindex, ishift, var_index);
                     hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index, t,
                                                      &entry);
                     hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                           matrix_type);
                     jElement_Edge[nElements_Edges]= rank;
                     nElements_Edges++;

                     hypre_SubtractIndex(var_index, kshift, var_index);
                     hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index, t,
                                                      &entry);
                     hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                           matrix_type);
                     jElement_Edge[nElements_Edges]= rank;
                     nElements_Edges++;

                     hypre_AddIndex(var_index, ishift, var_index);
                     hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index, t,
                                                      &entry);
                     hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                           matrix_type);
                     jElement_Edge[nElements_Edges]= rank;
                     nElements_Edges++;
                     break;
                  }

                  case 7: /* "|" z_Edge={(i,j,k),(i,j-1,k),(i-1,j-1,k),(i-1,j,k)}*/
                  {
                     hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex, t,
                                                      &entry);
                     hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                           matrix_type);
                     jElement_Edge[nElements_Edges]= rank;
                     nElements_Edges++;

                     hypre_SubtractIndex(cindex, jshift, var_index);
                     hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index, t,
                                                      &entry);
                     hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                           matrix_type);
                     jElement_Edge[nElements_Edges]= rank;
                     nElements_Edges++;

                     hypre_SubtractIndex(var_index, ishift, var_index);
                     hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index, t,
                                                      &entry);
                     hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                           matrix_type);
                     jElement_Edge[nElements_Edges]= rank;
                     nElements_Edges++;

                     hypre_AddIndex(var_index, jshift, var_index);
                     hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index, t,
                                                      &entry);
                     hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                           matrix_type);
                     jElement_Edge[nElements_Edges]= rank;
                     nElements_Edges++;
                     break;
                  }

               }   /* switch (var) */
            }      /* for (t= 0; t< Edge_nvars; t++) */
         }
         hypre_BoxLoop0End();
      }  /* hypre_ForBoxI(i, cboxes) */
   }     /* for (part= 0; part< nparts; part++) */

   if (ndim == 3)
   {
      HYPRE_IJMatrixSetValues(Element_Face, nElements, ncols_ElementFace,
                              (const HYPRE_Int*) iElement, (const HYPRE_Int*) jElement_Face,
                              (const double*) vals_ElementFace);
      HYPRE_IJMatrixAssemble((HYPRE_IJMatrix) Element_Face);
      hypre_TFree(ncols_ElementFace);
      hypre_TFree(jElement_Face);
      hypre_TFree(vals_ElementFace);
   }  /* if (ndim == 3) */

   HYPRE_IJMatrixSetValues(Element_Edge, nElements, ncols_ElementEdge,
                           (const HYPRE_Int*) iElement, (const HYPRE_Int*) jElement_Edge,
                           (const double*) vals_ElementEdge);
   HYPRE_IJMatrixAssemble((HYPRE_IJMatrix) Element_Edge);

   hypre_TFree(ncols_ElementEdge);
   hypre_TFree(iElement);
   hypre_TFree(jElement_Edge);
   hypre_TFree(vals_ElementEdge);

   /*-----------------------------------------------------------------------
    * edge_Edge, the actual interpolation matrix.
    * For each fine edge row, we need to know if it is a edge, 
    * boundary edge, or face edge. Knowing this allows us to determine the 
    * structure and weights of the interpolation matrix. 
    *
    * Scheme:A.Loop over contracted boxes of fine edge grid. 
    *          For each fine edge ijk,
    *     1) map it to a fine cell with the fine edge at the lower end
    *        of the box,e.g. x_edge[ijk] -> cell[i,j+1,k+1].
    *     2) coarsen the fine cell to obtain a coarse cell. Determine the
    *        location of the fine edge with respect to the coarse edges
    *        of this cell. Coarsening needed only when determining the
    *        column rank.
    *
    * Need to distinguish between 2-d and 3-d.
    *-----------------------------------------------------------------------*/

   /* count the row/col connections */
   iedgeEdge     = hypre_CTAlloc(HYPRE_Int, nedges);
   ncols_edgeEdge= hypre_CTAlloc(HYPRE_Int, nedges);

   /*-----------------------------------------------------------------------
    * loop first over the fedges aligning with the agglomerate coarse edges.
    * Will loop over the face & interior edges separately also. 
    * Since the weights for these edges will be used to determine the
    * weights along the face edges, we need to retrieve these computed
    * weights from vals_edgeEdge. Done by keeping a pointer of size nedges 
    * that points to the location of the weight:
    *          pointer[rank of edge]= index location where weight resides.
    *-----------------------------------------------------------------------*/
   j= 0;
   start_rank1= hypre_SStructGridStartRank(fgrid_edge);
   bdryedge_location= hypre_CTAlloc(HYPRE_Int, nedges);
   for (part= 0; part< nparts; part++)
   {
      p_fgrid= hypre_SStructGridPGrid(fgrid_edge, part);  /* edge grid */
      Edge_nvars= hypre_SStructPGridNVars(p_fgrid);
      Edge_vartypes= hypre_SStructPGridVarTypes(p_fgrid);

      /* note that fboxes are the contracted CELL boxes. Will get the correct
         variable grid extents. */
      fboxes= contract_fedgeBoxes[part];

      for (t= 0; t< Edge_nvars; t++)
      {
         var         = Edge_vartypes[t];
         var_fgrid   = hypre_SStructPGridVTSGrid(p_fgrid, var);
         box_array   = hypre_StructGridBoxes(var_fgrid);

         n_boxoffsets= ndim-1;
         boxoffset   = hypre_CTAlloc(hypre_Index, n_boxoffsets);
         suboffset   = hypre_CTAlloc(hypre_Index, n_boxoffsets);
         switch(var)
         {
            case 2: /* 2-d: x_face (vertical edges), stride=[rfactor[0],1,1] */
            {
               hypre_SetIndex(stride, rfactor[0], 1, 1);
               hypre_CopyIndex(varoffsets[2], var_index);

               /* boxoffset shrink in the i direction */
               hypre_SetIndex(boxoffset[0], rfactor[0]-1, 0, 0);
               hypre_SetIndex(suboffset[0], 1, 0, 0);

               /* extend loop_size by one in the stride direction */
               hypre_SetIndex(hi_index, 1, 0, 0);
               break;
            }

            case 3: /* 2-d: y_face (horizontal edges), stride=[1,rfactor[1],1] */
            {
               hypre_SetIndex(stride, 1, rfactor[1], 1);
               hypre_CopyIndex(varoffsets[3], var_index);

               /* boxoffset shrink in the j direction */
               hypre_SetIndex(boxoffset[0], 0, rfactor[1]-1, 0);
               hypre_SetIndex(suboffset[0], 0, 1, 0);

               /* extend loop_size by one in the stride direction */
               hypre_SetIndex(hi_index, 0, 1, 0);
               break;
            }

            case 5: /* 3-d: x_edge, stride=[1,rfactor[1],rfactor[2]] */
            {
               hypre_SetIndex(stride, 1, rfactor[1], rfactor[2]);
               hypre_CopyIndex(varoffsets[5], var_index);

               /* boxoffset shrink in the j & k directions */
               hypre_SetIndex(boxoffset[0], 0, rfactor[1]-1, 0);
               hypre_SetIndex(boxoffset[1], 0, 0, rfactor[2]-1);
               hypre_SetIndex(suboffset[0], 0, 1, 0);
               hypre_SetIndex(suboffset[1], 0, 0, 1);

               /* extend loop_size by one in the stride direction */
               hypre_SetIndex(hi_index, 0, 1, 1);
               break;
            }

            case 6: /* 3-d: y_edge, stride=[rfactor[0],1,rfactor[2]] */
            {
               hypre_SetIndex(stride, rfactor[0], 1, rfactor[2]);
               hypre_CopyIndex(varoffsets[6], var_index);

               /* boxoffset shrink in the i & k directions */
               hypre_SetIndex(boxoffset[0], rfactor[0]-1, 0, 0);
               hypre_SetIndex(boxoffset[1], 0, 0, rfactor[2]-1);
               hypre_SetIndex(suboffset[0], 1, 0, 0);
               hypre_SetIndex(suboffset[1], 0, 0, 1);

               /* extend loop_size by one in the stride direction */
               hypre_SetIndex(hi_index, 1, 0, 1);
               break;
            }

            case 7: /* 3-d: z_edge, stride=[rfactor[0],rfactor[1],1] */
            {
               hypre_SetIndex(stride, rfactor[0], rfactor[1], 1);
               hypre_CopyIndex(varoffsets[7], var_index);

               /* boxoffset shrink in the i & j directions */
               hypre_SetIndex(boxoffset[0], rfactor[0]-1, 0, 0);
               hypre_SetIndex(boxoffset[1], 0, rfactor[1]-1, 0);
               hypre_SetIndex(suboffset[0], 1, 0, 0);
               hypre_SetIndex(suboffset[1], 0, 1, 0);

               /* extend loop_size by one in the stride direction */
               hypre_SetIndex(hi_index, 1, 1, 0);
               break;
            }
         }

         hypre_ForBoxI(i, fboxes)
         {
            cellbox= hypre_BoxArrayBox(fboxes, i);

            /* vboxes inside the i'th cellbox */
            num_vboxes= n_CtoVbox[part][i];
            vboxnums  = CtoVboxnums[part][i];

            /* adjust the project cellbox to the variable box */
            hypre_CopyBox(cellbox, &copy_box);

            /* the adjusted variable box may be bigger than the actually
               variable box- variables that are shared may lead to smaller
               variable boxes than the SubtractIndex produces. If the box
               has to be decreased, then we decrease it by (rfactor[j]-1)
               in the appropriate direction.
               Check the location of the shifted lower box index. */
            for (k= 0; k< n_boxoffsets; k++)
            {
               hypre_SubtractIndex(hypre_BoxIMin(&copy_box), suboffset[k],
                                   findex);
               row_in= false;
               for (p= 0; p< num_vboxes[t]; p++)
               {
                  vbox= hypre_BoxArrayBox(box_array, vboxnums[t][p]);

                  if (hypre_IndexInBoxP(findex, vbox))
                  {
                     hypre_CopyIndex(findex, hypre_BoxIMin(&copy_box));
                     row_in= true;
                     break;
                  }
               }
               /* not in any vbox */
               if (!row_in)
               {
                  hypre_AddIndex(hypre_BoxIMin(&copy_box), boxoffset[k],
                                 hypre_BoxIMin(&copy_box));
               }
            }

            hypre_BoxGetSize(&copy_box, loop_size);
            hypre_StructMapFineToCoarse(loop_size, zero_index, stride,
                                        loop_size);

            /* extend the loop_size so that upper boundary of the box are reached. */
            hypre_AddIndex(loop_size, hi_index, loop_size);

            hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

            hypre_BoxLoop1Begin(loop_size,
                                &copy_box, start, stride, m);
#if 0
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,m,findex,k,entry,rank,j
#include "hypre_box_smp_forloop.h"
#else
            hypre_BoxLoopSetOneBlock();
#endif
            hypre_BoxLoop1For(loopi, loopj, loopk, m)
            {
               hypre_SetIndex(findex, loopi, loopj, loopk);
               for (k= 0; k< 3; k++)
               {
                  findex[k]*= stride[k];
               }
               hypre_AddIndex(findex, start, findex);

               hypre_SStructGridFindBoxManEntry(fgrid_edge, part, findex, t, &entry);
               hypre_SStructBoxManEntryGetGlobalRank(entry, findex, &rank, matrix_type);
               /* still row p may be outside the processor- check to make sure in */
               if ((rank <= fupper_ranks[part][var]) && (rank >= flower_ranks[part][var]))
               {
                  iedgeEdge[j]= rank;
                  ncols_edgeEdge[j]= 1;
                  bdryedge_location[rank-start_rank1]= j;
                  j++;
               }
            }
            hypre_BoxLoop1End(m);

         }   /* hypre_ForBoxI */
         hypre_TFree(boxoffset);
         hypre_TFree(suboffset);
      }  /* for (t= 0; t< nvars; t++) */
   }     /* for (part= 0; part< nparts; part++) */

   /*-----------------------------------------------------------------------
    * Record the row ranks for the face edges. Only for 3-d. 
    *
    * Loop over the face edges. 
    * Since the weights for these edges will be used to determine the
    * weights along the face edges, we need to retrieve these computed
    * weights form vals_edgeEdge. Done by keeping a pointer of size nedges 
    * that points to the location of the weight:
    *          pointer[rank of edge]= index location where weight resides.
    *-----------------------------------------------------------------------*/
   if (ndim == 3)
   {
      l= j;
      for (part= 0; part< nparts; part++)
      {
         p_fgrid= hypre_SStructGridPGrid(fgrid_edge, part);  /* edge grid */
         Edge_nvars= hypre_SStructPGridNVars(p_fgrid);
         Edge_vartypes= hypre_SStructPGridVarTypes(p_fgrid);

         /* note that fboxes are the contracted CELL boxes. Will get the correct
            variable grid extents. */
         fboxes= contract_fedgeBoxes[part];

         /* may need to shrink a given box in some boxoffset directions */
         boxoffset= hypre_TAlloc(hypre_Index, ndim);
         for (t= 0; t< ndim; t++)
         {
            hypre_ClearIndex(boxoffset[t]);
            hypre_IndexD(boxoffset[t], t)= rfactor[t]-1;
         }
           
         for (t= 0; t< Edge_nvars; t++)
         {
            var      = Edge_vartypes[t];
            var_fgrid=  hypre_SStructPGridVTSGrid(p_fgrid, var);
            box_array= hypre_StructGridBoxes(var_fgrid);

            /* to reduce comparison, take the switch outside of the loop */
            switch(var)
            {
               case 5:   
               {
                  /* 3-d x_edge, can be Y or Z_Face */
                  hypre_ForBoxI(i, fboxes)
                  {
                     cellbox= hypre_BoxArrayBox(fboxes, i);
      
                     /* vboxes inside the i'th cellbox */
                     num_vboxes= n_CtoVbox[part][i];
                     vboxnums  = CtoVboxnums[part][i];

                     /* adjust the contracted cellbox to the variable box */
                     hypre_CopyBox(cellbox, &copy_box);

                     /******************************************************
                      * Check the location of the shifted lower box index:
                      *         x_edge-> Z_Face & Y_Face:
                      *  Z_Face- contract in the z direction only if the
                      *          processor interface is in the z direction
                      *  Y_Face- contract in the y direction if the processor
                      *          interface is in the y direction.
                      ******************************************************/
                     hypre_SubtractIndex(hypre_BoxIMin(&copy_box), kshift,
                                         findex);
                     /* loop over all the vboxes to see if findex is inside */
                     row_in= false;
                     for (p= 0; p< num_vboxes[t]; p++)
                     {
                        vbox= hypre_BoxArrayBox(box_array, vboxnums[t][p]);
                        if (hypre_IndexInBoxP(findex, vbox))
                        {
                           hypre_CopyIndex(findex, hypre_BoxIMin(&copy_box));
                           row_in= true;
                           break;
                        }
                     }
                     /* not in any vbox */
                     if (!row_in)
                     {
                        hypre_AddIndex(hypre_BoxIMin(&copy_box), boxoffset[2],
                                       hypre_BoxIMin(&copy_box));
                     }
                     hypre_SubtractIndex(hypre_BoxIMin(&copy_box), jshift,
                                         hypre_BoxIMin(&copy_box));

                     hypre_BoxGetSize(&copy_box, loop_size);
                     hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                                 loop_size);

                     hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                     /* increase the loop_size by one in the Z plane direction */
                     loop_size[2]++;
                     hypre_BoxLoop1Begin(loop_size,
                                         &copy_box, start, rfactor, m);
#if 0
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,m,findex,p,var_index,n,entry,rank,j
#include "hypre_box_smp_forloop.h"
#else
                     hypre_BoxLoopSetOneBlock();
#endif
                     hypre_BoxLoop1For(loopi, loopj, loopk, m)
                     {
                        hypre_SetIndex(findex, loopi, loopj, loopk);
                        for (k= 0; k< 3; k++)
                        {
                           findex[k]*= rfactor[k];
                        }
                        hypre_AddIndex(findex, start, findex);

                        /************************************************************
                         * Loop over the Z_Face x_edges.
                         ************************************************************/
                        for (p= 0; p< rfactor[0]; p++)
                        {
                           hypre_CopyIndex(findex, var_index);
                           var_index[0]+= p;
                           for (n= 1; n< rfactor[1]; n++)
                           {
                              var_index[1]++;
                              hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              /* still row rank may be outside the processor */
                              if ((rank <= fupper_ranks[part][var]) &&
                                  (rank >= flower_ranks[part][var]))
                              {
                                 iedgeEdge[j]= rank;

                                 /* Z_Face. Two coarse Edge connections. */
                                 ncols_edgeEdge[j]= 2;
                                 j++;

                                 /* record index location */
                                 bdryedge_location[rank-start_rank1]= l;
                                 l+= 2;  /* two weight values */
                              }
                           }  /* for (n= 1; n< rfactor[1]; n++) */
                        }     /* for (p= 0; p< rfactor[0]; p++) */
                     }
                     hypre_BoxLoop1End(m);

                     /* Y_Face */
                     hypre_CopyBox(cellbox, &copy_box);
                     hypre_SubtractIndex(hypre_BoxIMin(&copy_box), jshift,
                                         findex);
                     /* loop over all the vboxes to see if findex is inside */
                     row_in= false;
                     for (p= 0; p< num_vboxes[t]; p++)
                     {
                        vbox= hypre_BoxArrayBox(box_array, vboxnums[t][p]);
                        if (hypre_IndexInBoxP(findex, vbox))
                        {
                           hypre_CopyIndex(findex, hypre_BoxIMin(&copy_box));
                           row_in= true;
                           break;
                        }
                     }
                     /* not in any vbox */
                     if (!row_in)
                     {
                        hypre_AddIndex(hypre_BoxIMin(&copy_box), boxoffset[1],
                                       hypre_BoxIMin(&copy_box));
                     }
                     hypre_SubtractIndex(hypre_BoxIMin(&copy_box), kshift,
                                         hypre_BoxIMin(&copy_box));

                     hypre_BoxGetSize(&copy_box, loop_size);
                     hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                                 loop_size);

                     hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);
                     loop_size[1]++;
                     hypre_BoxLoop1Begin(loop_size,
                                         &copy_box, start, rfactor, m);
#if 0
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,m,findex,k,p,var_index,n,entry,rank,j,l
#include "hypre_box_smp_forloop.h"
#else
                     hypre_BoxLoopSetOneBlock();
#endif
                     hypre_BoxLoop1For(loopi, loopj, loopk, m)
                     {
                        hypre_SetIndex(findex, loopi, loopj, loopk);
                        for (k= 0; k< 3; k++)
                        {
                           findex[k]*= rfactor[k];
                        }
                        hypre_AddIndex(findex, start, findex);

                        /************************************************************
                         * Loop over the Y_Face x_edges.
                         ************************************************************/
                        for (p= 0; p< rfactor[0]; p++)
                        {
                           hypre_CopyIndex(findex, var_index);
                           var_index[0]+= p;
                           for (n= 1; n< rfactor[2]; n++)
                           {
                              var_index[2]++;
                              hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              if ((rank <= fupper_ranks[part][var]) &&
                                  (rank >= flower_ranks[part][var]))
                              {
                                 iedgeEdge[j]= rank;

                                 /* Y_Face. Two coarse Edge connections. */
                                 ncols_edgeEdge[j]= 2;
                                 j++;

                                 /* record index location */
                                 bdryedge_location[rank-start_rank1]= l;
                                 l+= 2;
                              }
                           }  /* for (n= 1; n< rfactor[2]; n++) */
                        }     /* for (p= 0; p< rfactor[0]; p++) */
                     }
                     hypre_BoxLoop1End(m);
                  }  /* hypre_ForBoxI(i, fboxes) */

                  break;
               }

               case 6:   
               {
                  /* 3-d y_edge, can be X or Z_Face */
                  hypre_ForBoxI(i, fboxes)
                  {
                     cellbox= hypre_BoxArrayBox(fboxes, i);
      
                     /* vboxes inside the i'th cellbox */
                     num_vboxes= n_CtoVbox[part][i];
                     vboxnums  = CtoVboxnums[part][i];

                     /* adjust the project cellbox to the variable box */
                     hypre_CopyBox(cellbox, &copy_box);

                     /******************************************************
                      * Check the location of the shifted lower box index:
                      *         y_edge-> X_Face & Z_Face:
                      *  Z_Face- contract in the z direction only if the
                      *          processor interface is in the z direction
                      *  X_Face- contract in the x direction if the processor
                      *          interface is in the x direction.
                      ******************************************************/
                     hypre_SubtractIndex(hypre_BoxIMin(&copy_box), kshift,
                                         findex);
                     /* loop over all the vboxes to see if findex is inside */
                     row_in= false;
                     for (p= 0; p< num_vboxes[t]; p++)
                     {
                        vbox= hypre_BoxArrayBox(box_array, vboxnums[t][p]);
                        if (hypre_IndexInBoxP(findex, vbox))
                        {
                           hypre_CopyIndex(findex, hypre_BoxIMin(&copy_box));
                           row_in= true;
                           break;
                        }
                     }
                     /* not in any vbox */
                     if (!row_in)
                     {
                        hypre_AddIndex(hypre_BoxIMin(&copy_box), boxoffset[2],
                                       hypre_BoxIMin(&copy_box));
                     }
                     hypre_SubtractIndex(hypre_BoxIMin(&copy_box), ishift,
                                         hypre_BoxIMin(&copy_box));

                     hypre_BoxGetSize(&copy_box, loop_size);
                     hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                                 loop_size);

                     hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                     /* reset and then increase the loop_size by one in the Z_Face direction */
                     loop_size[2]++;
                     hypre_BoxLoop1Begin(loop_size,
                                         &copy_box, start, rfactor, m);
#if 0
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,m,findex,k,p,var_index,n,entry,rank,j,l
#include "hypre_box_smp_forloop.h"
#else
                     hypre_BoxLoopSetOneBlock();
#endif
                     hypre_BoxLoop1For(loopi, loopj, loopk, m)
                     {
                        hypre_SetIndex(findex, loopi, loopj, loopk);

                        for (k= 0; k< 3; k++)
                        {
                           findex[k]*= rfactor[k];
                        }
                        hypre_AddIndex(findex, start, findex);

                        /* Z_Face */
                        for (p= 0; p< rfactor[1]; p++)
                        {
                           hypre_CopyIndex(findex, var_index);
                           var_index[1]+= p;
                           for (n= 1; n< rfactor[0]; n++)
                           {
                              var_index[0]++;
                              hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              if ((rank <= fupper_ranks[part][var]) &&
                                  (rank >= flower_ranks[part][var]))
                              {
                                 iedgeEdge[j]= rank;

                                 /* Z_Face. Two coarse Edge connections. */
                                 ncols_edgeEdge[j]= 2;
                                 j++;

                                 /* record index location */
                                 bdryedge_location[rank-start_rank1]= l;
                                 l+= 2;
                              }
                           }  /* for (n= 1; n< rfactor[0]; n++) */
                        }     /* for (p= 0; p< rfactor[1]; p++) */
                     }
                     hypre_BoxLoop1End(m);

                     /* X_Face */
                     hypre_CopyBox(cellbox, &copy_box);
                     hypre_SubtractIndex(hypre_BoxIMin(&copy_box), ishift,
                                         findex);
                     /* loop over all the vboxes to see if findex is inside */
                     row_in= false;
                     for (p= 0; p< num_vboxes[t]; p++)
                     {
                        vbox= hypre_BoxArrayBox(box_array, vboxnums[t][p]);
                        if (hypre_IndexInBoxP(findex, vbox))
                        {
                           hypre_CopyIndex(findex, hypre_BoxIMin(&copy_box));
                           row_in= true;
                           break;
                        }
                     }
                     /* not in any vbox */
                     if (!row_in)
                     {
                        hypre_AddIndex(hypre_BoxIMin(&copy_box), boxoffset[0],
                                       hypre_BoxIMin(&copy_box));
                     }
                     hypre_SubtractIndex(hypre_BoxIMin(&copy_box), kshift,
                                         hypre_BoxIMin(&copy_box));

                     hypre_BoxGetSize(&copy_box, loop_size);
                     hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                                 loop_size);

                     hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                     loop_size[0]++;
                     hypre_BoxLoop1Begin(loop_size,
                                         &copy_box, start, rfactor, m);
#if 0
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,m,findex,k,p,var_index,n,entry,rank,j,l
#include "hypre_box_smp_forloop.h"
#else
                     hypre_BoxLoopSetOneBlock();
#endif
                     hypre_BoxLoop1For(loopi, loopj, loopk, m)
                     {
                        hypre_SetIndex(findex, loopi, loopj, loopk);
                        for (k= 0; k< 3; k++)
                        {
                           findex[k]*= rfactor[k];
                        }
                        hypre_AddIndex(findex, start, findex);

                        /*****************************************************
                         * Loop over the X_Face y_edges.
                         *****************************************************/
                        for (p= 0; p< rfactor[1]; p++)
                        {
                           hypre_CopyIndex(findex, var_index);
                           var_index[1]+= p;
                           for (n= 1; n< rfactor[2]; n++)
                           {
                              var_index[2]++;
                              hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              if ((rank <= fupper_ranks[part][var]) &&
                                  (rank >= flower_ranks[part][var]))
                              {
                                 iedgeEdge[j]= rank;
                                 /* X_Face. Two coarse Edge connections. */
                                 ncols_edgeEdge[j]= 2;
                                 j++;

                                 /* record index location */
                                 bdryedge_location[rank-start_rank1]= l;
                                 l+= 2;
                              }
                           }  /* for (n= 1; n< rfactor[2]; n++) */
                        }     /* for (p= 0; p< rfactor[1]; p++) */
                     }
                     hypre_BoxLoop1End(m);
                  }  /* hypre_ForBoxI(i, fboxes) */

                  break;
               }

               case 7:   
               {
                  /* 3-d z_edge, can be interior, X or Y_Face, or Z_Edge */
                  hypre_ForBoxI(i, fboxes)
                  {
                     cellbox= hypre_BoxArrayBox(fboxes, i);

                     /* vboxes inside the i'th cellbox */
                     num_vboxes= n_CtoVbox[part][i];
                     vboxnums  = CtoVboxnums[part][i];
      
                     /* adjust the project cellbox to the variable box */
                     hypre_CopyBox(cellbox, &copy_box);

                     /******************************************************
                      * Check the location of the shifted lower box index:
                      *         z_edge-> X_Face & Y_Face:
                      *  X_Face- contract in the x direction if the processor
                      *          interface is in the x direction.
                      *  Y_Face- contract in the y direction if the processor
                      *          interface is in the y direction.
                      ******************************************************/
                     hypre_SubtractIndex(hypre_BoxIMin(&copy_box), ishift,
                                         findex);
                     /* loop over all the vboxes to see if findex is inside */
                     row_in= false;
                     for (p= 0; p< num_vboxes[t]; p++)
                     {
                        vbox= hypre_BoxArrayBox(box_array, vboxnums[t][p]);
                        if (hypre_IndexInBoxP(findex, vbox))
                        {
                           hypre_CopyIndex(findex, hypre_BoxIMin(&copy_box));
                           row_in= true;
                           break;
                        }
                     }
                     /* not in any vbox */
                     if (!row_in)
                     {
                        hypre_AddIndex(hypre_BoxIMin(&copy_box), boxoffset[0],
                                       hypre_BoxIMin(&copy_box));
                     }
                     hypre_SubtractIndex(hypre_BoxIMin(&copy_box), jshift,
                                         hypre_BoxIMin(&copy_box));

                     hypre_BoxGetSize(&copy_box, loop_size);
                     hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                                 loop_size);

                     hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                     /* increase the loop_size by one in the X_Face direction */
                     loop_size[0]++;
                     hypre_BoxLoop1Begin(loop_size,
                                         &copy_box, start, rfactor, m);
#if 0
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,m,findex,k,p,var_index,n,entry,rank,j,l
#include "hypre_box_smp_forloop.h"
#else
                     hypre_BoxLoopSetOneBlock();
#endif
                     hypre_BoxLoop1For(loopi, loopj, loopk, m)
                     {
                        hypre_SetIndex(findex, loopi, loopj, loopk);

                        for (k= 0; k< 3; k++)
                        {
                           findex[k]*= rfactor[k];
                        }

                        hypre_AddIndex(findex, start, findex);

                        /******************************************************
                         * Loop over the X_Face z_edges.
                         ******************************************************/
                        for (p= 0; p< rfactor[2]; p++)
                        {
                           hypre_CopyIndex(findex, var_index);
                           var_index[2]+= p;
                           for (n= 1; n< rfactor[1]; n++)
                           {
                              var_index[1]++;
                              hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              if ((rank <= fupper_ranks[part][var]) &&
                                  (rank >= flower_ranks[part][var]))
                              {
                                 iedgeEdge[j]= rank;

                                 /* X_Face. Two coarse Edge connections. */
                                 ncols_edgeEdge[j]= 2;
                                 j++;

                                 /* record index location */
                                 bdryedge_location[rank-start_rank1]= l;
                                 l+= 2;
                              }
                           }  /* for (n= 1; n< rfactor[1]; n++) */
                        }     /* for (p= 0; p< rfactor[2]; p++) */
                     }
                     hypre_BoxLoop1End(m);

                     /* Y_Face */
                     hypre_CopyBox(cellbox, &copy_box);

                     hypre_SubtractIndex(hypre_BoxIMin(&copy_box), jshift,
                                         findex);
                     /* loop over all the vboxes to see if findex is inside */
                     row_in= false;
                     for (p= 0; p< num_vboxes[t]; p++)
                     {
                        vbox= hypre_BoxArrayBox(box_array, vboxnums[t][p]);
                        if (hypre_IndexInBoxP(findex, vbox))
                        {
                           hypre_CopyIndex(findex, hypre_BoxIMin(&copy_box));
                           row_in= true;
                           break;
                        }
                     }
                     /* not in any vbox */
                     if (!row_in)
                     {
                        hypre_AddIndex(hypre_BoxIMin(&copy_box), boxoffset[1],
                                       hypre_BoxIMin(&copy_box));
                     }
                     hypre_SubtractIndex(hypre_BoxIMin(&copy_box), ishift,
                                         hypre_BoxIMin(&copy_box));

                     hypre_BoxGetSize(&copy_box, loop_size);
                     hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                                 loop_size);
                     hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                     loop_size[1]++;
                     hypre_BoxLoop1Begin(loop_size,
                                         &copy_box, start, rfactor, m);
#if 0
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,m,findex,k,p,var_index,n,entry,rank,j,l
#include "hypre_box_smp_forloop.h"
#else
                     hypre_BoxLoopSetOneBlock();
#endif
                     hypre_BoxLoop1For(loopi, loopj, loopk, m)
                     {
                        hypre_SetIndex(findex, loopi, loopj, loopk);

                        for (k= 0; k< 3; k++)
                        {
                           findex[k]*= rfactor[k];
                        }

                        hypre_AddIndex(findex, start, findex);
                        /****************************************************
                         * Loop over the Y_Face z_edges.
                         ****************************************************/
                        for (p= 0; p< rfactor[2]; p++)
                        {
                           hypre_CopyIndex(findex, var_index);
                           var_index[2]+= p;
                           for (n= 1; n< rfactor[0]; n++)
                           {
                              var_index[0]++;
                              hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              if ((rank <= fupper_ranks[part][var]) &&
                                  (rank >= flower_ranks[part][var]))
                              {
                                 iedgeEdge[j]= rank;

                                 /* Y_Face. Two coarse Edge connections. */
                                 ncols_edgeEdge[j]= 2;
                                 j++;

                                 /* record index location */
                                 bdryedge_location[rank-start_rank1]= l;
                                 l+= 2;
                              }
                           }  /* for (n= 1; n< rfactor[0]; n++) */
                        }     /* for (p= 0; p< rfactor[2]; p++) */
                     }
                     hypre_BoxLoop1End(m);
                  }  /* hypre_ForBoxI(i, fboxes) */

                  break;
               }

            }  /* switch */
         }     /* for (t= 0; t< Edge_nvars; t++) */

         hypre_TFree(boxoffset);
      }  /* for (part= 0; part< nparts; part++) */
   }     /* if (ndim == 3) */

   for (part= 0; part< nparts; part++)
   {
      p_fgrid= hypre_SStructGridPGrid(fgrid_edge, part);  /* edge grid */
      Edge_nvars= hypre_SStructPGridNVars(p_fgrid);
      Edge_vartypes= hypre_SStructPGridVarTypes(p_fgrid);

      /* note that fboxes are the contracted CELL boxes. Will get the correct
         variable grid extents. */
      fboxes= contract_fedgeBoxes[part];

      for (t= 0; t< Edge_nvars; t++)
      {
         var      = Edge_vartypes[t];
         var_fgrid= hypre_SStructPGridVTSGrid(p_fgrid, var);
         box_array= hypre_StructGridBoxes(var_fgrid);

         /* to reduce comparison, take the switch outside of the loop */
         switch(var)
         {
            case 2:  
            {
               /* 2-d x_face = x_edge, can be interior */
               hypre_ForBoxI(i, fboxes)
               {
                  cellbox= hypre_BoxArrayBox(fboxes, i);
                  vbox   = hypre_BoxArrayBox(box_array, i);
      
                  /* adjust the contracted cellbox to the variable box */
                  hypre_CopyBox(cellbox, &copy_box);
                  hypre_SubtractIndex(hypre_BoxIMin(&copy_box), varoffsets[var], 
                                      hypre_BoxIMin(&copy_box));

                  hypre_BoxGetSize(&copy_box, loop_size);
                  hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                              loop_size);
                  hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                  hypre_BoxLoop1Begin(loop_size,
                                      &copy_box, start, rfactor, m);
#if 0
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,m,findex,k,p,var_index,n,entry,rank,j
#include "hypre_box_smp_forloop.h"
#else
                  hypre_BoxLoopSetOneBlock();
#endif
                  hypre_BoxLoop1For(loopi, loopj, loopk, m)
                  {
                     hypre_SetIndex(findex, loopi, loopj, loopk);
                     for (k= 0; k< 3; k++)
                     {
                        findex[k]*= rfactor[k];
                     }
                     hypre_AddIndex(findex, start, findex);

                     /* get interior edges */
                     for (p= 1; p< rfactor[0]; p++)
                     {
                        hypre_CopyIndex(findex, var_index);
                        var_index[0]+= p;
                        for (n= 0; n< rfactor[1]; n++)
                        {
                           hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                            t, &entry);
                           hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                 matrix_type);
                           iedgeEdge[j]= rank;

                           /* lies interior of Face. Four coarse Edge connection. */
                           ncols_edgeEdge[j]= 4;
                           j++;

                           var_index[1]++;
                        }  /* for (n= 0; n< rfactor[1]; n++) */
                     }     /* for (p= 1; p< rfactor[0]; p++) */

                  }
                  hypre_BoxLoop1End(m);
               }  /* hypre_ForBoxI(i, fboxes) */
               break;
            }

            case 3:   
            {
               /* 2-d y_face = y_edge, can be interior */
               hypre_ForBoxI(i, fboxes)
               {
                  cellbox= hypre_BoxArrayBox(fboxes, i);
                  vbox   = hypre_BoxArrayBox(box_array, i);
      
                  /* adjust the project cellbox to the variable box */
                  hypre_CopyBox(cellbox, &copy_box);
                  hypre_SubtractIndex(hypre_BoxIMin(&copy_box), varoffsets[var], 
                                      hypre_BoxIMin(&copy_box));

                  hypre_BoxGetSize(&copy_box, loop_size);
                  hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                              loop_size);
                  hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                  hypre_BoxLoop1Begin(loop_size,
                                      &copy_box, start, rfactor, m);
#if 0
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,m,findex,k,p,var_index,n,entry,rank,j
#include "hypre_box_smp_forloop.h"
#else
                  hypre_BoxLoopSetOneBlock();
#endif
                  hypre_BoxLoop1For(loopi, loopj, loopk, m)
                  {
                     hypre_SetIndex(findex, loopi, loopj, loopk);

                     for (k= 0; k< 3; k++)
                     {
                        findex[k]*= rfactor[k];
                     }

                     hypre_AddIndex(findex, start, findex);

                     /* get interior edges */
                     for (p= 1; p< rfactor[1]; p++)
                     {
                        hypre_CopyIndex(findex, var_index);
                        var_index[1]+= p;
                        for (n= 0; n< rfactor[0]; n++)
                        {
                           hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                            t, &entry);
                           hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                 matrix_type);
                           iedgeEdge[j]= rank;

                           /* lies interior of Face. Four coarse Edge connection. */
                           ncols_edgeEdge[j]= 4;
                           j++;

                           var_index[0]++;
                        }  /* for (n= 0; n< rfactor[0]; n++) */
                     }     /* for (p= 1; p< rfactor[1]; p++) */
                  }
                  hypre_BoxLoop1End(m);
               }  /* hypre_ForBoxI(i, fboxes) */
               break;
            }

            case 5:   
            {
               /* 3-d x_edge, can be only interior */
               hypre_ForBoxI(i, fboxes)
               {
                  cellbox= hypre_BoxArrayBox(fboxes, i);
                  vbox   = hypre_BoxArrayBox(box_array, i);
      
                  /* adjust the project cellbox to the variable box */
                  hypre_CopyBox(cellbox, &copy_box);
                  hypre_SubtractIndex(hypre_BoxIMin(&copy_box), varoffsets[var], 
                                      hypre_BoxIMin(&copy_box));

                  hypre_BoxGetSize(&copy_box, loop_size);
                  hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                              loop_size);
                  hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                  hypre_BoxLoop1Begin(loop_size,
                                      &copy_box, start, rfactor, m);
#if 0
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,m,findex,k,p,var_index,n,entry,rank,j
#include "hypre_box_smp_forloop.h"
#else
                  hypre_BoxLoopSetOneBlock();
#endif
                  hypre_BoxLoop1For(loopi, loopj, loopk, m)
                  {
                     hypre_SetIndex(findex, loopi, loopj, loopk);
                     for (k= 0; k< 3; k++)
                     {
                        findex[k]*= rfactor[k];
                     }
                     hypre_AddIndex(findex, start, findex);

                     /* get interior edges */
                     for (p= 1; p< rfactor[2]; p++)
                     {
                        hypre_CopyIndex(findex, var_index);
                        var_index[2]+= p;
                        for (n= 1; n< rfactor[1]; n++)
                        {
                           var_index[1]++;
                           for (k= 0; k< rfactor[0]; k++)
                           {
                              hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              iedgeEdge[j]= rank;

                              /* Interior. Twelve coarse Edge connections. */
                              ncols_edgeEdge[j]= 12;
                              j++;

                              var_index[0]++;
                           }  /* for (k= 0; k< rfactor[0]; k++) */
                          
                           /* reset var_index[0] to the initial index for next k loop */
                           var_index[0]-= rfactor[0];  

                        }  /* for (n= 1; n< rfactor[1]; n++) */

                        /* reset var_index[1] to the initial index for next n loop */
                        var_index[1]-= (rfactor[1]-1);  
                     }  /* for (p= 1; p< rfactor[2]; p++) */

                  }
                  hypre_BoxLoop1End(m);
               }  /* hypre_ForBoxI(i, fboxes) */
               break;
            }

            case 6:   
            {
               /* 3-d y_edge, can be only interior */
               hypre_ForBoxI(i, fboxes)
               {
                  cellbox= hypre_BoxArrayBox(fboxes, i);
                  vbox   = hypre_BoxArrayBox(box_array, i);
      
                  /* adjust the contract cellbox to the variable box */
                  hypre_CopyBox(cellbox, &copy_box);
                  hypre_SubtractIndex(hypre_BoxIMin(&copy_box), varoffsets[var], 
                                      hypre_BoxIMin(&copy_box));

                  hypre_BoxGetSize(&copy_box, loop_size);
                  hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                              loop_size);
                  hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                  hypre_BoxLoop1Begin(loop_size,
                                      &copy_box, start, rfactor, m);
#if 0
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,m,findex,k,p,var_index,n,entry,rank,j
#include "hypre_box_smp_forloop.h"
#else
                  hypre_BoxLoopSetOneBlock();
#endif
                  hypre_BoxLoop1For(loopi, loopj, loopk, m)
                  {
                     hypre_SetIndex(findex, loopi, loopj, loopk);
                     for (k= 0; k< 3; k++)
                     {
                        findex[k]*= rfactor[k];
                     }
                     hypre_AddIndex(findex, start, findex);

                     /* get interior edges */
                     for (p= 1; p< rfactor[2]; p++)
                     {
                        hypre_CopyIndex(findex, var_index);
                        var_index[2]+= p;
                        for (n= 1; n< rfactor[0]; n++)
                        {
                           var_index[0]++;
                           for (k= 0; k< rfactor[1]; k++)
                           {
                              hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              iedgeEdge[j]= rank;

                              /* Interior. Twelve coarse Edge connections. */
                              ncols_edgeEdge[j]= 12;
                              j++;

                              var_index[1]++;
                           }  /* for (k= 0; k< rfactor[1]; k++) */

                           /* reset var_index[1] to the initial index for next k loop */
                           var_index[1]-= rfactor[1];

                        }  /* for (n= 1; n< rfactor[0]; n++) */

                        /* reset var_index[0] to the initial index for next n loop */
                        var_index[0]-= (rfactor[0]-1);
                     }  /* for (p= 1; p< rfactor[2]; p++) */

                  }
                  hypre_BoxLoop1End(m);
               }  /* hypre_ForBoxI(i, fboxes) */

               break;
            }

            case 7:   
            {
               /* 3-d z_edge, can be only interior */
               hypre_ForBoxI(i, fboxes)
               {
                  cellbox= hypre_BoxArrayBox(fboxes, i);
                  vbox   = hypre_BoxArrayBox(box_array, i);
      
                  /* adjust the contracted cellbox to the variable box */
                  hypre_CopyBox(cellbox, &copy_box);
                  hypre_SubtractIndex(hypre_BoxIMin(&copy_box), varoffsets[var], 
                                      hypre_BoxIMin(&copy_box));

                  hypre_BoxGetSize(&copy_box, loop_size);
                  hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                              loop_size);
                  hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                  hypre_BoxLoop1Begin(loop_size,
                                      &copy_box, start, rfactor, m);
#if 0
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,m,findex,k,p,var_index,n,entry,rank,j
#include "hypre_box_smp_forloop.h"
#else
                  hypre_BoxLoopSetOneBlock();
#endif
                  hypre_BoxLoop1For(loopi, loopj, loopk, m)
                  {
                     hypre_SetIndex(findex, loopi, loopj, loopk);
                     for (k= 0; k< 3; k++)
                     {
                        findex[k]*= rfactor[k];
                     }
                     hypre_AddIndex(findex, start, findex);

                     /* get interior edges */
                     for (p= 1; p< rfactor[1]; p++)
                     {
                        hypre_CopyIndex(findex, var_index);
                        var_index[1]+= p;
                        for (n= 1; n< rfactor[0]; n++)
                        {
                           var_index[0]++;
                           for (k= 0; k< rfactor[2]; k++)
                           {
                              hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              iedgeEdge[j]= rank;

                              /* Interior. Twelve coarse Edge connections. */
                              ncols_edgeEdge[j]= 12;
                              j++;

                              var_index[2]++;
                           }  /* for (k= 0; k< rfactor[2]; k++) */

                           /* reset var_index[2] to the initial index for next k loop */
                           var_index[2]-= rfactor[2];

                        }  /* for (n= 1; n< rfactor[0]; n++) */

                        /* reset var_index[0] to the initial index for next n loop */
                        var_index[0]-= (rfactor[0]-1);
                     }  /* for (p= 1; p< rfactor[1]; p++) */

                  }
                  hypre_BoxLoop1End(m);
               }  /* hypre_ForBoxI(i, fboxes) */
               break;
            }

         }  /* switch */
      }     /* for (t= 0; t< Edge_nvars; t++) */
   }        /* for (part= 0; part< nparts; part++) */

   k= 0;
   j= 0;
   for (i= 0; i< nedges; i++)
   {
      if (ncols_edgeEdge[i])
      {
         k+= ncols_edgeEdge[i];
         j++;
      }
   }
   vals_edgeEdge = hypre_CTAlloc(double, k);
   jedge_Edge    = hypre_CTAlloc(HYPRE_Int, k);
   size1         = j;

   /*********************************************************************
    * Fill up the edge_Edge interpolation matrix. Interpolation weights
    * are determined differently for each type of fine edges.
    *
    * fedge_on_CEdge: use geometric interpolation, i.e., length of
    * edge ratio.
    *
    * fedge_on_agglomerate_face: box mg approach. Collapse the like
    * variable stencil connections of the given face. Weighted linear
    * interpolation of the fedge_on_CEdge values.
    *
    * fedge_in_agglomerate_interior: amge.
    *********************************************************************/

   /* loop over fedges aligning with the agglomerate coarse edges first. */
   k= 0;
   for (part= 0; part< nparts; part++)
   {
      p_fgrid= hypre_SStructGridPGrid(fgrid_edge, part);  /* edge grid */
      Edge_nvars= hypre_SStructPGridNVars(p_fgrid);
      Edge_vartypes= hypre_SStructPGridVarTypes(p_fgrid);
      p_cgrid= hypre_SStructGridPGrid(cgrid_edge, part);  /* Edge grid */

      /* note that fboxes are the contracted CELL boxes. Will get the correct
         variable grid extents. */
      fboxes= contract_fedgeBoxes[part];

      for (t= 0; t< Edge_nvars; t++)
      {
         var      = Edge_vartypes[t];
         var_fgrid= hypre_SStructPGridVTSGrid(p_fgrid, var);
         box_array= hypre_StructGridBoxes(var_fgrid);

         n_boxoffsets= ndim-1;
         boxoffset   = hypre_CTAlloc(hypre_Index, n_boxoffsets);
         suboffset   = hypre_CTAlloc(hypre_Index, n_boxoffsets);
         switch(var)
         {
            case 2: /* 2-d: x_face (vertical edges), stride=[rfactor[0],1,1] 
                       fCedge_ratio= 1.0/rfactor[1] */
            {
               hypre_SetIndex(stride, rfactor[0], 1, 1);
               component_stride= 1; /*stride vertically, component y= 1.*/
               fCedge_ratio= 1.0/rfactor[1];

               /* boxoffset shrink in the i direction */
               hypre_SetIndex(boxoffset[0], rfactor[0]-1, 0, 0);
               hypre_SetIndex(suboffset[0], 1, 0, 0);

               /* extend loop_size by one in the stride direction */
               hypre_SetIndex(hi_index, 1, 0, 0);
               break;
            }

            case 3: /* 2-d: y_face (horizontal edges), stride=[1,rfactor[1],1] 
                       fCedge_ratio= 1.0/rfactor[0] */
            {
               hypre_SetIndex(stride, 1, rfactor[1], 1);
               component_stride= 0; /*stride horizontally, component x= 0.*/
               fCedge_ratio= 1.0/rfactor[0];

               /* boxoffset shrink in the j direction */
               hypre_SetIndex(boxoffset[0], 0, rfactor[1]-1, 0);
               hypre_SetIndex(suboffset[0], 0, 1, 0);

               /* extend loop_size by one in the stride direction */
               hypre_SetIndex(hi_index, 0, 1, 0);
               break;
            }

            case 5: /* 3-d: x_edge, stride=[1,rfactor[1],rfactor[2]] 
                       fCedge_ratio= 1.0/rfactor[0] */
            {
               hypre_SetIndex(stride, 1, rfactor[1], rfactor[2]);
               component_stride= 0; /*stride x direction, component x= 0.*/
               fCedge_ratio= 1.0/rfactor[0];

               /* boxoffset shrink in the j & k directions */
               hypre_SetIndex(boxoffset[0], 0, rfactor[1]-1, 0);
               hypre_SetIndex(boxoffset[1], 0, 0, rfactor[2]-1);
               hypre_SetIndex(suboffset[0], 0, 1, 0);
               hypre_SetIndex(suboffset[1], 0, 0, 1);

               /* extend loop_size by one in the stride direction */
               hypre_SetIndex(hi_index, 0, 1, 1);
               break;
            }

            case 6: /* 3-d: y_edge, stride=[rfactor[0],1,rfactor[2]] 
                       fCedge_ratio= 1.0/rfactor[1] */
            {
               hypre_SetIndex(stride, rfactor[0], 1, rfactor[2]);
               component_stride= 1; /*stride y direction, component y= 1.*/
               fCedge_ratio= 1.0/rfactor[1];

               /* boxoffset shrink in the i & k directions */
               hypre_SetIndex(boxoffset[0], rfactor[0]-1, 0, 0);
               hypre_SetIndex(boxoffset[1], 0, 0, rfactor[2]-1);
               hypre_SetIndex(suboffset[0], 1, 0, 0);
               hypre_SetIndex(suboffset[1], 0, 0, 1);

               /* extend loop_size by one in the stride direction */
               hypre_SetIndex(hi_index, 1, 0, 1);
               break;
            }

            case 7: /* 3-d: z_edge, stride=[rfactor[0],rfactor[1],1]
                       fCedge_ratio= 1.0/rfactor[2] */
            {
               hypre_SetIndex(stride, rfactor[0], rfactor[1], 1);
               component_stride= 2; /*stride z direction, component z= 2.*/
               fCedge_ratio= 1.0/rfactor[2];
                                                                     
               /* boxoffset shrink in the i & j directions */
               hypre_SetIndex(boxoffset[0], rfactor[0]-1, 0, 0);
               hypre_SetIndex(boxoffset[1], 0, rfactor[1]-1, 0);
               hypre_SetIndex(suboffset[0], 1, 0, 0);
               hypre_SetIndex(suboffset[1], 0, 1, 0);

               /* extend loop_size by one in the stride direction */
               hypre_SetIndex(hi_index, 1, 1, 0);
               break;
            }
         }

         hypre_ForBoxI(i, fboxes)
         {
            cellbox= hypre_BoxArrayBox(fboxes, i);

            /* vboxes inside the i'th cellbox */
            num_vboxes= n_CtoVbox[part][i];
            vboxnums  = CtoVboxnums[part][i];

            hypre_CopyIndex(Edge_cstarts[part][i], cstart);

            /* adjust the contracted cellbox to the variable box.
               Note that some of the fboxes may be skipped because they
               vanish. */
            hypre_CopyBox(cellbox, &copy_box);

            for (j= 0; j< n_boxoffsets; j++)
            {
               hypre_SubtractIndex(hypre_BoxIMin(&copy_box), suboffset[j],
                                   findex);
               row_in= false;
               for (p= 0; p< num_vboxes[t]; p++)
               {
                  vbox= hypre_BoxArrayBox(box_array, vboxnums[t][p]);

                  if (hypre_IndexInBoxP(findex, vbox))
                  {
                     hypre_CopyIndex(findex, hypre_BoxIMin(&copy_box));
                     row_in= true;
                     break;
                  }
               }
               /* not in any vbox */
               if (!row_in)
               {
                  hypre_AddIndex(hypre_BoxIMin(&copy_box), boxoffset[j],
                                 hypre_BoxIMin(&copy_box));

                  /* also modify cstart */
                  hypre_AddIndex(boxoffset[j], one_index, boxoffset[j]);
                  hypre_StructMapFineToCoarse(boxoffset[j], zero_index, rfactor,
                                              boxoffset[j]);
                  hypre_AddIndex(cstart, boxoffset[j], cstart);
               }
            }

            hypre_BoxGetSize(&copy_box, loop_size);
            hypre_StructMapFineToCoarse(loop_size, zero_index, stride,
                                        loop_size);

            /* extend the loop_size so that upper boundary of the box are reached. */
            hypre_AddIndex(loop_size, hi_index, loop_size);

            hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

            /* note that the correct cbox corresponding to this non-vanishing
               fbox is used. */

            hypre_BoxLoop1Begin(loop_size,
                                &copy_box, start, stride, m);
#if 0
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,m,findex,j,entry,cindex,var_index,rank,k
#include "hypre_box_smp_forloop.h"
#else
            hypre_BoxLoopSetOneBlock();
#endif
            hypre_BoxLoop1For(loopi, loopj, loopk, m)
            {
               hypre_SetIndex(findex, loopi, loopj, loopk);
               for (j= 0; j< 3; j++)
               {
                  findex[j]*= stride[j];
               }

               hypre_AddIndex(findex, start, findex);
               hypre_SStructGridFindBoxManEntry(fgrid_edge, part, findex, t, &entry);
               hypre_SStructBoxManEntryGetGlobalRank(entry, findex, &j, matrix_type);

               /* still row p may be outside the processor- check to make sure in */
               if ((j <= fupper_ranks[part][var]) && (j >= flower_ranks[part][var]))
               {
                  hypre_SubtractIndex(findex, start, findex);

                  /* determine where the edge lies- coarsening required. */
                  hypre_StructMapFineToCoarse(findex, zero_index, rfactor,
                                              cindex);
                  hypre_AddIndex(cindex, cstart, cindex);
                  hypre_AddIndex(findex, start, findex);

                  /* lies on coarse Edge. Coarse Edge connection:
                     var_index= cindex - subtract_index.*/
                  hypre_SubtractIndex(cindex, varoffsets[var], var_index);

                  hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                   t, &entry);
                  hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                        matrix_type);
                  jedge_Edge[k]= rank;
                  vals_edgeEdge[k]= fCedge_ratio;

                  k++;
               }
            }
            hypre_BoxLoop1End(m);
         }   /* hypre_ForBoxI */
         hypre_TFree(boxoffset);
         hypre_TFree(suboffset);
      }  /* for (t= 0; t< nvars; t++) */
   }     /* for (part= 0; part< nparts; part++) */

   /* generate the face interpolation weights/info. Only for 3-d */
   if (ndim == 3)
   {
      /* Allocate memory to arrays for the tridiagonal system & solutions.
         Take the maximum size needed. */
      i= rfactor[0]-1;
      for (j= 1; j< ndim; j++)
      {
         if (i < (rfactor[j]-1))
         {
            i= rfactor[j]-1;
         }
      }
      upper= hypre_CTAlloc(double, i);
      lower= hypre_CTAlloc(double, i);
      diag = hypre_CTAlloc(double, i);
      face_w1= hypre_CTAlloc(double, i);
      face_w2= hypre_CTAlloc(double, i);
      off_proc_flag= hypre_CTAlloc(HYPRE_Int, i+1);

      for (part= 0; part< nparts; part++)
      {
         p_fgrid= hypre_SStructGridPGrid(fgrid_edge, part);  /* edge grid */
         Edge_nvars= hypre_SStructPGridNVars(p_fgrid);
         Edge_vartypes= hypre_SStructPGridVarTypes(p_fgrid);
         p_cgrid= hypre_SStructGridPGrid(cgrid_edge, part);  /* Edge grid */

         /* note that fboxes are the contracted CELL boxes. Will get the correct
            variable grid extents. */
         fboxes= contract_fedgeBoxes[part];

         /* may need to shrink a given box in some boxoffset directions */
         boxoffset= hypre_TAlloc(hypre_Index, ndim);
         for (t= 0; t< ndim; t++)
         {
            hypre_ClearIndex(boxoffset[t]);
            hypre_IndexD(boxoffset[t], t)= rfactor[t]-1;
         }

         for (t= 0; t< Edge_nvars; t++)
         {
            var      = Edge_vartypes[t];
            var_fgrid=  hypre_SStructPGridVTSGrid(p_fgrid, var);
            box_array= hypre_StructGridBoxes(var_fgrid);
            switch(var)
            {
               case 5:
               {
                  /* 3-d x_edge, can be Y or Z_Face */
                  fCedge_ratio= 1.0/rfactor[0];
                  hypre_ForBoxI(i, fboxes)
                  {
                     cellbox= hypre_BoxArrayBox(fboxes, i);

                     /* vboxes inside the i'th cellbox */
                     num_vboxes= n_CtoVbox[part][i];
                     vboxnums  = CtoVboxnums[part][i];

                     hypre_CopyIndex(Edge_cstarts[part][i], cstart);

                     /* adjust the project cellbox to the variable box */
                     hypre_CopyBox(cellbox, &copy_box);

                     /******************************************************
                      * Check the location of the shifted lower box index:
                      *         x_edge-> Z_Face & Y_Face:
                      *  Z_Face- contract in the z direction only if the
                      *          processor interface is in the z direction
                      *  Y_Face- contract in the y direction if the processor
                      *          interface is in the y direction.
                      ******************************************************/
                     hypre_SubtractIndex(hypre_BoxIMin(&copy_box), kshift,
                                         findex);
                     /* loop over all the vboxes to see if findex is inside */
                     row_in= false;
                     for (p= 0; p< num_vboxes[t]; p++)
                     {
                        vbox= hypre_BoxArrayBox(box_array, vboxnums[t][p]);
                        if (hypre_IndexInBoxP(findex, vbox))
                        {
                           hypre_CopyIndex(findex, hypre_BoxIMin(&copy_box));
                           row_in= true;
                           break;
                        }
                     }
                     /* not in any vbox */
                     if (!row_in)
                     {
                        hypre_AddIndex(hypre_BoxIMin(&copy_box), boxoffset[2],
                                       hypre_BoxIMin(&copy_box));

                        /* modify cstart */
                        hypre_AddIndex(cstart, kshift, cstart);
                     }
                     hypre_SubtractIndex(hypre_BoxIMin(&copy_box), jshift,
                                         hypre_BoxIMin(&copy_box));

                     hypre_BoxGetSize(&copy_box, loop_size);
                     hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                                 loop_size);
                     hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                     /* increase the loop_size by one in the Z plane direction */
                     loop_size[2]++;
                     hypre_BoxLoop1Begin(loop_size,
                                         &copy_box, start, rfactor, m);
#if 0
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,m,findex,cindex,l,var_index,entry,rank2,rank,p,n,face_w1,face_w2,off_proc_flag,stencil_vals,lower,diag,upper,k
#include "hypre_box_smp_forloop.h"
#else
                     hypre_BoxLoopSetOneBlock();
#endif
                     hypre_BoxLoop1For(loopi, loopj, loopk, m)
                     {
                        hypre_SetIndex(findex, loopi, loopj, loopk);

                        /* because of rfactor striding, cindex= findex. But adjust
                           by cstart to get actually coarse edge. */
                        hypre_CopyIndex(findex, cindex);
                        hypre_AddIndex(cindex, cstart, cindex);

                        /* Will need the actual fine indices. */
                        for (l= 0; l< ndim; l++)
                        {
                           findex[l]*= rfactor[l];
                        }
                        hypre_AddIndex(findex, start, findex);

                        hypre_SubtractIndex(cindex, kshift, var_index);
                        hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                         t, &entry);
                        hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank2,
                                                              matrix_type);

                        hypre_SubtractIndex(var_index, jshift, var_index);
                        hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                         t, &entry);
                        hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                              matrix_type);

                        /* loop over the strips of x_edges making up the Z_Face and
                           create the tridiagonal systems by collapsing the stencils. */
                        for (p= 0; p< rfactor[0]; p++)
                        {
                           /* create the rhs's for the tridiagonal system. Require
                              find the ranks at the endpt's of the strip. */
                           for (n= 0; n< rfactor[1]-1; n++)
                           {
                              face_w1[n]= 0.0;
                              face_w2[n]= 0.0;
                           }

                           /******************************************************
                            * grab the already computed lower-end edge weight.
                            * These are bdry agglomerate wgts that are pre-determined
                            * so that no communication is needed.
                            ******************************************************/

                           /* lower-end and upper-end edge weights */
                           face_w1[0]= fCedge_ratio;
                           face_w2[rfactor[1]-2]= fCedge_ratio;

                           /******************************************************
                            * create tridiagonal matrix.
                            * x_edge for Z_Face: collapse_dir= 2, stencil_dir= 1
                            ******************************************************/
                           hypre_CopyIndex(findex, var_index);
                           var_index[0]+= p;
                           for (n= 1; n< rfactor[1]; n++)
                           {
                              var_index[1]++;
                              off_proc_flag[n]= 
                                 hypre_CollapseStencilToStencil(Aee,
                                                                fgrid_edge,
                                                                part,
                                                                t,
                                                                var_index,
                                                                2,
                                                                1,
                                                                &stencil_vals);
                              /* put extracted stencil_vals into tridiagonal matrix */
                              lower[n-1]= stencil_vals[0];
                              diag[n-1] = stencil_vals[1];
                              upper[n-1]= stencil_vals[2];
                              hypre_TFree(stencil_vals);
                           }

                           /* solve systems to get weights. Must adjust face_w's so
                              that the stencil entry contributes. */
                           face_w1[0]*= -lower[0];
                           face_w2[rfactor[1]-2]*= -upper[rfactor[1]-2];
                           hypre_TriDiagSolve(diag, upper, lower, face_w1, rfactor[1]-1);
                           hypre_TriDiagSolve(diag, upper, lower, face_w2, rfactor[1]-1);

                           /* place weights into vals_edgeEdge */
                           for (n= 1; n< rfactor[1]; n++)
                           {
                              if (!off_proc_flag[n])  /* off_proc_flag= 1 if offproc */
                              {
                                 jedge_Edge[k]= rank;
                                 vals_edgeEdge[k]= face_w1[n-1]; /* lower end connection */
                                 k++;

                                 jedge_Edge[k]= rank2;
                                 vals_edgeEdge[k]= face_w2[n-1]; /* upper end connection */
                                 k++;
                              }
                           }
                        }  /* for (p= 0; p< rfactor[0]; p++) */
                     }
                     hypre_BoxLoop1End(m);

                     /* Y_Face */
                     hypre_CopyIndex(Edge_cstarts[part][i], cstart);
                     hypre_CopyBox(cellbox, &copy_box);
                     hypre_SubtractIndex(hypre_BoxIMin(&copy_box), jshift,
                                         findex);
                     /* loop over all the vboxes to see if findex is inside */
                     row_in= false;
                     for (p= 0; p< num_vboxes[t]; p++)
                     {
                        vbox= hypre_BoxArrayBox(box_array, vboxnums[t][p]);
                        if (hypre_IndexInBoxP(findex, vbox))
                        {
                           hypre_CopyIndex(findex, hypre_BoxIMin(&copy_box));
                           row_in= true;
                           break;
                        }
                     }
                     /* not in any vbox */
                     if (!row_in)
                     {
                        hypre_AddIndex(hypre_BoxIMin(&copy_box), boxoffset[1],
                                       hypre_BoxIMin(&copy_box));

                        /* modify cstart */
                        hypre_AddIndex(cstart, jshift, cstart);
                     }
                     hypre_SubtractIndex(hypre_BoxIMin(&copy_box), kshift,
                                         hypre_BoxIMin(&copy_box));

                     hypre_BoxGetSize(&copy_box, loop_size);
                     hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                                 loop_size);
                     hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                     loop_size[1]++;
                     hypre_BoxLoop1Begin(loop_size,
                                         &copy_box, start, rfactor, m);
#if 0
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,m,findex,cindex,l,var_index,entry,rank2,rank,p,n,face_w1,face_w2,off_proc_flag,stencil_vals,lower,diag,upper,k
#include "hypre_box_smp_forloop.h"
#else
                     hypre_BoxLoopSetOneBlock();
#endif
                     hypre_BoxLoop1For(loopi, loopj, loopk, m)
                     {
                        hypre_SetIndex(findex, loopi, loopj, loopk);

                        /* because of rfactor striding, cindex= findex. But adjust
                           by cstart to get actually coarse edge. */
                        hypre_CopyIndex(findex, cindex);
                        hypre_AddIndex(cindex, cstart, cindex);

                        /* Will need the actual fine indices. */ 
                        for (l= 0; l< ndim; l++)
                        {
                           findex[l]*= rfactor[l];
                        }
                        hypre_AddIndex(findex, start, findex);

                        /******************************************************
                         * Y_Face. Two coarse Edge connections. 
                         * x_Edge (i,j-1,k), (i,j-1,k-1)
                         ******************************************************/
                        hypre_SubtractIndex(cindex, jshift, var_index);
                        hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                         t, &entry);
                        hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank2,
                                                              matrix_type);

                        hypre_SubtractIndex(var_index, kshift, var_index);
                        hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                         t, &entry);
                        hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                              matrix_type);

                        /* loop over the strips of x_edges making up the Y_Face and
                           create the tridiagonal systems by collapsing the stencils. */
                        for (p= 0; p< rfactor[0]; p++)
                        {
                           /* create the rhs's for the tridiagonal system. Require
                              find the ranks at the endpt's of the strip. */
                           for (n= 0; n< rfactor[2]-1; n++)
                           {
                              face_w1[n]= 0.0;
                              face_w2[n]= 0.0;
                           }

                           /* lower-end and upper-end edge weights */
                           face_w1[0]= fCedge_ratio;
                           face_w2[rfactor[2]-2]= fCedge_ratio;

                           /******************************************************
                            * create tridiagonal matrix.
                            * x_edge for Y_Face: collapse_dir= 1, stencil_dir= 2
                            ******************************************************/
                           hypre_CopyIndex(findex, var_index);
                           var_index[0]+= p;
                           for (n= 1; n< rfactor[2]; n++)
                           {
                              var_index[2]++;
                              off_proc_flag[n]= 
                                 hypre_CollapseStencilToStencil(Aee,
                                                                fgrid_edge,
                                                                part,
                                                                t,
                                                                var_index,
                                                                1,
                                                                2,
                                                                &stencil_vals);
                              /* put extracted stencil_vals into tridiagonal matrix */
                              lower[n-1]= stencil_vals[0];
                              diag[n-1] = stencil_vals[1];
                              upper[n-1]= stencil_vals[2];
                              hypre_TFree(stencil_vals);
                           }

                           /* solve systems to get weights. Must adjust face_w's so
                              that the stencil entry contributes. */
                           face_w1[0]*= -lower[0];
                           face_w2[rfactor[2]-2]*= -upper[rfactor[2]-2];
                           hypre_TriDiagSolve(diag, upper, lower, face_w1, rfactor[2]-1);
                           hypre_TriDiagSolve(diag, upper, lower, face_w2, rfactor[2]-1);

                           /* place weights into vals_edgeEdge */
                           for (n= 1; n< rfactor[2]; n++)
                           {
                              if (!off_proc_flag[n])  /* off_proc_flag= 1 if offproc */
                              {
                                 jedge_Edge[k]= rank;
                                 vals_edgeEdge[k]= face_w1[n-1]; /* lower end connection */
                                 k++;

                                 jedge_Edge[k]= rank2;
                                 vals_edgeEdge[k]= face_w2[n-1]; /* upper end connection */
                                 k++;
                              }
                           }
                        }  /* for (p= 0; p< rfactor[0]; p++) */

                     }
                     hypre_BoxLoop1End(m);
                  }  /* hypre_ForBoxI(i, fboxes) */
                  break;
               }

               case 6:
               {
                  /* 3-d y_edge, can be X or Z_Face */
                  fCedge_ratio= 1.0/rfactor[1];
                  hypre_ForBoxI(i, fboxes)
                  {
                     cellbox= hypre_BoxArrayBox(fboxes, i);

                     /* vboxes inside the i'th cellbox */
                     num_vboxes= n_CtoVbox[part][i];
                     vboxnums  = CtoVboxnums[part][i];

                     hypre_CopyIndex(Edge_cstarts[part][i], cstart);

                     /* adjust the project cellbox to the variable box */
                     hypre_CopyBox(cellbox, &copy_box);

                     /******************************************************
                      * Check the location of the shifted lower box index:
                      *         y_edge-> X_Face & Z_Face:
                      *  Z_Face- contract in the z direction only if the
                      *          processor interface is in the z direction
                      *  X_Face- contract in the x direction if the processor
                      *          interface is in the x direction.
                      ******************************************************/
                     /* Z_Face */
                     hypre_SubtractIndex(hypre_BoxIMin(&copy_box), kshift,
                                         findex);
                     /* loop over all the vboxes to see if findex is inside */
                     row_in= false;
                     for (p= 0; p< num_vboxes[t]; p++)
                     {
                        vbox= hypre_BoxArrayBox(box_array, vboxnums[t][p]);
                        if (hypre_IndexInBoxP(findex, vbox))
                        {
                           hypre_CopyIndex(findex, hypre_BoxIMin(&copy_box));
                           row_in= true;
                           break;
                        }
                     }
                     /* not in any vbox */
                     if (!row_in)
                     {
                        hypre_AddIndex(hypre_BoxIMin(&copy_box), boxoffset[2],
                                       hypre_BoxIMin(&copy_box));

                        /* modify cstart */
                        hypre_AddIndex(cstart, kshift, cstart);
                     }
                     hypre_SubtractIndex(hypre_BoxIMin(&copy_box), ishift,
                                         hypre_BoxIMin(&copy_box));

                     hypre_BoxGetSize(&copy_box, loop_size);
                     hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                                 loop_size);
                     hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                     /* increase the loop_size by one in the Z plane direction */
                     loop_size[2]++;
                     hypre_BoxLoop1Begin(loop_size,
                                         &copy_box, start, rfactor, m);
#if 0
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,m,findex,cindex,l,var_index,entry,rank2,rank,p,n,face_w1,face_w2,off_proc_flag,stencil_vals,lower,diag,upper,k
#include "hypre_box_smp_forloop.h"
#else
                     hypre_BoxLoopSetOneBlock();
#endif
                     hypre_BoxLoop1For(loopi, loopj, loopk, m)
                     {
                        hypre_SetIndex(findex, loopi, loopj, loopk);

                        /* because of rfactor striding, cindex= findex. But adjust
                           by cstart to get actually coarse edge. */
                        hypre_CopyIndex(findex, cindex);
                        hypre_AddIndex(cindex, cstart, cindex);

                        /* Will need the actual fine indices. */
                        for (l= 0; l< ndim; l++)
                        {
                           findex[l]*= rfactor[l];
                        }
                        hypre_AddIndex(findex, start, findex);

                        hypre_SubtractIndex(cindex, kshift, var_index);
                        hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                         t, &entry);
                        hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank2,
                                                              matrix_type);

                        hypre_SubtractIndex(var_index, ishift, var_index);
                        hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                         t, &entry);
                        hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                              matrix_type);

                        /* loop over the strips of y_edges making up the Z_Face and
                           create the tridiagonal systems by collapsing the stencils. */
                        for (p= 0; p< rfactor[1]; p++)
                        {
                           /* create the rhs's for the tridiagonal system. Require
                              find the ranks at the endpt's of the strip. */
                           for (n= 0; n< rfactor[0]-1; n++)
                           {
                              face_w1[n]= 0.0;
                              face_w2[n]= 0.0;
                           }

                           /* lower-end and upper-end edge weights */
                           face_w1[0]= fCedge_ratio;
                           face_w2[rfactor[0]-2]= fCedge_ratio;

                           /******************************************************
                            * create tridiagonal matrix.
                            * y_edge for Z_Face: collapse_dir= 2, stencil_dir= 0
                            ******************************************************/
                           hypre_CopyIndex(findex, var_index);
                           var_index[1]+= p;
                           for (n= 1; n< rfactor[0]; n++)
                           {
                              var_index[0]++;
                              off_proc_flag[n]= 
                                 hypre_CollapseStencilToStencil(Aee,
                                                                fgrid_edge,
                                                                part,
                                                                t,
                                                                var_index,
                                                                2,
                                                                0,
                                                                &stencil_vals);
                              /* put extracted stencil_vals into tridiagonal matrix */
                              lower[n-1]= stencil_vals[0];
                              diag[n-1] = stencil_vals[1];
                              upper[n-1]= stencil_vals[2];
                              hypre_TFree(stencil_vals);
                           }

                           /* solve systems to get weights. Must adjust face_w's so
                              that the stencil entry contributes. */
                           face_w1[0]*= -lower[0];
                           face_w2[rfactor[0]-2]*= -upper[rfactor[0]-2];
                           hypre_TriDiagSolve(diag, upper, lower, face_w1, rfactor[0]-1);
                           hypre_TriDiagSolve(diag, upper, lower, face_w2, rfactor[0]-1);

                           /* place weights into vals_edgeEdge */
                           for (n= 1; n< rfactor[0]; n++)
                           {
                              if (!off_proc_flag[n])  /* off_proc_flag= 1 if offproc */
                              {
                                 jedge_Edge[k]= rank;
                                 vals_edgeEdge[k]= face_w1[n-1]; /* lower end connection */
                                 k++;

                                 jedge_Edge[k]= rank2;
                                 vals_edgeEdge[k]= face_w2[n-1]; /* upper end connection */
                                 k++;
                              }
                           }
                        }  /* for (p= 0; p< rfactor[1]; p++) */
                     }
                     hypre_BoxLoop1End(m);

                     /* X_Face */
                     hypre_CopyBox(cellbox, &copy_box);
                     hypre_CopyIndex(Edge_cstarts[part][i], cstart);

                     hypre_SubtractIndex(hypre_BoxIMin(&copy_box), ishift,
                                         findex);
                     /* loop over all the vboxes to see if findex is inside */
                     row_in= false;
                     for (p= 0; p< num_vboxes[t]; p++)
                     {
                        vbox= hypre_BoxArrayBox(box_array, vboxnums[t][p]);
                        if (hypre_IndexInBoxP(findex, vbox))
                        {
                           hypre_CopyIndex(findex, hypre_BoxIMin(&copy_box));
                           row_in= true;
                           break;
                        }
                     }
                     /* not in any vbox */
                     if (!row_in)
                     {
                        hypre_AddIndex(hypre_BoxIMin(&copy_box), boxoffset[0],
                                       hypre_BoxIMin(&copy_box));

                        /* modify cstart */
                        hypre_AddIndex(cstart, ishift, cstart);
                     }
                     hypre_SubtractIndex(hypre_BoxIMin(&copy_box), kshift,
                                         hypre_BoxIMin(&copy_box));

                     hypre_BoxGetSize(&copy_box, loop_size);
                     hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                                 loop_size);
                     hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                     loop_size[0]++;
                     hypre_BoxLoop1Begin(loop_size,
                                         &copy_box, start, rfactor, m);
#if 0
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,m,findex,cindex,l,var_index,entry,rank2,rank,p,n,face_w1,face_w2,off_proc_flag,stencil_vals,lower,diag,upper,k
#include "hypre_box_smp_forloop.h"
#else
                     hypre_BoxLoopSetOneBlock();
#endif
                     hypre_BoxLoop1For(loopi, loopj, loopk, m)
                     {
                        hypre_SetIndex(findex, loopi, loopj, loopk);

                        /* because of rfactor striding, cindex= findex. But adjust
                           by cstart to get actually coarse edge. */
                        hypre_CopyIndex(findex, cindex);
                        hypre_AddIndex(cindex, cstart, cindex);

                        /* Will need the actual fine indices. */
                        for (l= 0; l< ndim; l++)
                        {
                           findex[l]*= rfactor[l];
                        }
                        hypre_AddIndex(findex, start, findex);

                        /******************************************************
                         * X_Face. Two coarse Edge connections. 
                         * y_Edge (i-1,j,k), (i-1,j,k-1)
                         ******************************************************/
                        hypre_SubtractIndex(cindex, ishift, var_index);
                        hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                         t, &entry);
                        hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank2,
                                                              matrix_type);

                        hypre_SubtractIndex(var_index, kshift, var_index);
                        hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                         t, &entry);
                        hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                              matrix_type);

                        /* loop over the strips of y_edges making up the X_Face and
                           create the tridiagonal systems by collapsing the stencils. */
                        for (p= 0; p< rfactor[1]; p++)
                        {
                           /* create the rhs's for the tridiagonal system. Require
                              find the ranks at the endpt's of the strip. */
                           for (n= 0; n< rfactor[2]-1; n++)
                           {
                              face_w1[n]= 0.0;
                              face_w2[n]= 0.0;
                           }

                           /* lower-end and upper-end edge weights */
                           face_w1[0]= fCedge_ratio;
                           face_w2[rfactor[0]-2]= fCedge_ratio;

                           /******************************************************
                            * create tridiagonal matrix.
                            * y_edge for X_Face: collapse_dir= 0, stencil_dir= 2
                            ******************************************************/
                           hypre_CopyIndex(findex, var_index);
                           var_index[1]+= p;
                           for (n= 1; n< rfactor[2]; n++)
                           {
                              var_index[2]++;
                              off_proc_flag[n]= 
                                 hypre_CollapseStencilToStencil(Aee,
                                                                fgrid_edge,
                                                                part,
                                                                t,
                                                                var_index,
                                                                0,
                                                                2,
                                                                &stencil_vals);
                              /* put extracted stencil_vals into tridiagonal matrix */
                              lower[n-1]= stencil_vals[0];
                              diag[n-1] = stencil_vals[1];
                              upper[n-1]= stencil_vals[2];
                              hypre_TFree(stencil_vals);
                           }

                           /* solve systems to get weights. Must adjust face_w's so
                              that the stencil entry contributes. */
                           face_w1[0]*= -lower[0];
                           face_w2[rfactor[2]-2]*= -upper[rfactor[2]-2];
                           hypre_TriDiagSolve(diag, upper, lower, face_w1, rfactor[2]-1);
                           hypre_TriDiagSolve(diag, upper, lower, face_w2, rfactor[2]-1);

                           /* place weights into vals_edgeEdge */
                           for (n= 1; n< rfactor[2]; n++)
                           {
                              if (!off_proc_flag[n])  /* off_proc_flag= 1 if offproc */
                              {
                                 jedge_Edge[k]= rank;
                                 vals_edgeEdge[k]= face_w1[n-1]; /* lower end connection */
                                 k++;

                                 jedge_Edge[k]= rank2;
                                 vals_edgeEdge[k]= face_w2[n-1]; /* upper end connection */
                                 k++;
                              }
                           }
                        }  /* for (p= 0; p< rfactor[1]; p++) */

                     }
                     hypre_BoxLoop1End(m);
                  }  /* hypre_ForBoxI(i, fboxes) */
                  break;
               }

               case 7:
               {
                  /* 3-d z_edge, can be X or Y_Face */
                  fCedge_ratio= 1.0/rfactor[2];
                  hypre_ForBoxI(i, fboxes)
                  {
                     cellbox= hypre_BoxArrayBox(fboxes, i);

                     /* vboxes inside the i'th cellbox */
                     num_vboxes= n_CtoVbox[part][i];
                     vboxnums  = CtoVboxnums[part][i];

                     hypre_CopyIndex(Edge_cstarts[part][i], cstart);

                     /* adjust the project cellbox to the variable box */
                     hypre_CopyBox(cellbox, &copy_box);

                     /******************************************************
                      * Check the location of the shifted lower box index:
                      *         z_edge-> X_Face & Y_Face:
                      *  X_Face- contract in the x direction if the processor
                      *          interface is in the x direction.
                      *  Y_Face- contract in the y direction if the processor
                      *          interface is in the y direction.
                      ******************************************************/
                     hypre_SubtractIndex(hypre_BoxIMin(&copy_box), ishift,
                                         findex);
                     /* loop over all the vboxes to see if findex is inside */
                     row_in= false;
                     for (p= 0; p< num_vboxes[t]; p++)
                     {
                        vbox= hypre_BoxArrayBox(box_array, vboxnums[t][p]);
                        if (hypre_IndexInBoxP(findex, vbox))
                        {
                           hypre_CopyIndex(findex, hypre_BoxIMin(&copy_box));
                           row_in= true;
                           break;
                        }
                     }
                     /* not in any vbox */
                     if (!row_in)
                     {
                        hypre_AddIndex(hypre_BoxIMin(&copy_box), boxoffset[0],
                                       hypre_BoxIMin(&copy_box));

                        /* modify cstart */
                        hypre_AddIndex(cstart, ishift, cstart);
                     }
                     hypre_SubtractIndex(hypre_BoxIMin(&copy_box), jshift,
                                         hypre_BoxIMin(&copy_box));

                     hypre_BoxGetSize(&copy_box, loop_size);
                     hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                                 loop_size);

                     hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                     /* increase the loop_size by one in the X plane direction */
                     loop_size[0]++;
                     hypre_BoxLoop1Begin(loop_size,
                                         &copy_box, start, rfactor, m);
#if 0
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,m,findex,cindex,l,var_index,entry,rank2,rank,p,n,face_w1,face_w2,off_proc_flag,stencil_vals,lower,diag,upper,k
#include "hypre_box_smp_forloop.h"
#else
                     hypre_BoxLoopSetOneBlock();
#endif
                     hypre_BoxLoop1For(loopi, loopj, loopk, m)
                     {
                        hypre_SetIndex(findex, loopi, loopj, loopk);

                        /* because of rfactor striding, cindex= findex. But adjust
                           by cstart to get actually coarse edge. */
                        hypre_CopyIndex(findex, cindex);
                        hypre_AddIndex(cindex, cstart, cindex);

                        /* Will need the actual fine indices. */
                        for (l= 0; l< ndim; l++)
                        {
                           findex[l]*= rfactor[l];
                        }
                        hypre_AddIndex(findex, start, findex);

                        hypre_SubtractIndex(cindex, ishift, var_index);
                        hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                         t, &entry);
                        hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank2,
                                                              matrix_type);

                        hypre_SubtractIndex(var_index, jshift, var_index);
                        hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                         t, &entry);
                        hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                              matrix_type);

                        /* loop over the strips of z_edges making up the X_Face and
                           create the tridiagonal systems by collapsing the stencils. */
                        for (p= 0; p< rfactor[2]; p++)
                        {
                           /* create the rhs's for the tridiagonal system. Require
                              find the ranks at the endpt's of the strip. */
                           for (n= 0; n< rfactor[1]-1; n++)
                           {
                              face_w1[n]= 0.0;
                              face_w2[n]= 0.0;
                           }

                           /* lower-end and upper-end edge weights */
                           face_w1[0]= fCedge_ratio;
                           face_w2[rfactor[1]-2]= fCedge_ratio;

                           /******************************************************
                            * create tridiagonal matrix.
                            * z_edge for X_Face: collapse_dir= 0, stencil_dir= 1
                            ******************************************************/
                           hypre_CopyIndex(findex, var_index);
                           var_index[2]+= p;
                           for (n= 1; n< rfactor[1]; n++)
                           {
                              var_index[1]++;
                              off_proc_flag[n]= 
                                 hypre_CollapseStencilToStencil(Aee,
                                                                fgrid_edge,
                                                                part,
                                                                t,
                                                                var_index,
                                                                0,
                                                                1,
                                                                &stencil_vals);
                              /* put extracted stencil_vals into tridiagonal matrix */
                              lower[n-1]= stencil_vals[0];
                              diag[n-1] = stencil_vals[1];
                              upper[n-1]= stencil_vals[2];
                              hypre_TFree(stencil_vals);
                           }

                           /* solve systems to get weights. Must adjust face_w's so
                              that the stencil entry contributes. */
                           face_w1[0]*= -lower[0];
                           face_w2[rfactor[1]-2]*= -upper[rfactor[1]-2];
                           hypre_TriDiagSolve(diag, upper, lower, face_w1, rfactor[1]-1);
                           hypre_TriDiagSolve(diag, upper, lower, face_w2, rfactor[1]-1);

                           /* place weights into vals_edgeEdge */
                           for (n= 1; n< rfactor[1]; n++)
                           {
                              if (!off_proc_flag[n])  /* off_proc_flag= 1 if offproc */
                              {
                                 jedge_Edge[k]= rank;
                                 vals_edgeEdge[k]= face_w1[n-1]; /* lower end connection */
                                 k++;

                                 jedge_Edge[k]= rank2;
                                 vals_edgeEdge[k]= face_w2[n-1]; /* upper end connection */
                                 k++;
                              }
                           }
                        }  /* for (p= 0; p< rfactor[2]; p++) */
                     }
                     hypre_BoxLoop1End(m);

                     /* Y_Face */
                     hypre_CopyBox(cellbox, &copy_box);
                     hypre_CopyIndex(Edge_cstarts[part][i], cstart);

                     hypre_SubtractIndex(hypre_BoxIMin(&copy_box), jshift,
                                         findex);
                     /* loop over all the vboxes to see if findex is inside */
                     row_in= false;
                     for (p= 0; p< num_vboxes[t]; p++)
                     {
                        vbox= hypre_BoxArrayBox(box_array, vboxnums[t][p]);
                        if (hypre_IndexInBoxP(findex, vbox))
                        {
                           hypre_CopyIndex(findex, hypre_BoxIMin(&copy_box));
                           row_in= true;
                           break;
                        }
                     }
                     /* not in any vbox */
                     if (!row_in)
                     {
                        hypre_AddIndex(hypre_BoxIMin(&copy_box), boxoffset[1],
                                       hypre_BoxIMin(&copy_box));
                        /* modify cstart */
                        hypre_AddIndex(cstart, jshift, cstart);
                     }
                     hypre_SubtractIndex(hypre_BoxIMin(&copy_box), ishift,
                                         hypre_BoxIMin(&copy_box));

                     hypre_BoxGetSize(&copy_box, loop_size);
                     hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                                 loop_size);
                     hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                     loop_size[1]++;
                     hypre_BoxLoop1Begin(loop_size,
                                         &copy_box, start, rfactor, m);
#if 0
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,m,findex,cindex,l,var_index,entry,rank2,rank,p,n,face_w1,face_w2,off_proc_flag,stencil_vals,lower,diag,upper,k
#include "hypre_box_smp_forloop.h"
#else
                     hypre_BoxLoopSetOneBlock();
#endif
                     hypre_BoxLoop1For(loopi, loopj, loopk, m)
                     {
                        hypre_SetIndex(findex, loopi, loopj, loopk);

                        /* because of rfactor striding, cindex= findex. But adjust
                           by cstart to get actually coarse edge. */
                        hypre_CopyIndex(findex, cindex);
                        hypre_AddIndex(cindex, cstart, cindex);

                        /* Will need the actual fine indices. */
                        for (l= 0; l< ndim; l++)
                        {
                           findex[l]*= rfactor[l];
                        }
                        hypre_AddIndex(findex, start, findex);

                        /**********************************************************
                         * Y_Face (i,j-1,k). Two like-var coarse Edge connections.
                         * z_Edge (i,j-1,k), (i-1,j-1,k)
                         **********************************************************/
                        hypre_SubtractIndex(cindex, jshift, var_index);
                        hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                         t, &entry);
                        hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank2,
                                                              matrix_type);

                        hypre_SubtractIndex(var_index, ishift, var_index);
                        hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                         t, &entry);
                        hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                              matrix_type);

                        /* loop over the strips of y_edges making up the Y_Face and
                           create the tridiagonal systems by collapsing the stencils. */
                        for (p= 0; p< rfactor[2]; p++)
                        {
                           /* create the rhs's for the tridiagonal system. Require
                              find the ranks at the endpt's of the strip. */
                           for (n= 0; n< rfactor[0]-1; n++)
                           {
                              face_w1[n]= 0.0;
                              face_w2[n]= 0.0;
                           }

                           /* lower-end and upper-end edge weights */
                           face_w1[0]= fCedge_ratio;
                           face_w2[rfactor[0]-2]= fCedge_ratio;

                           /******************************************************
                            * create tridiagonal matrix.
                            * z_edge for Y_Face: collapse_dir= 1, stencil_dir= 0
                            ******************************************************/
                           hypre_CopyIndex(findex, var_index);
                           var_index[2]+= p;
                           for (n= 1; n< rfactor[0]; n++)
                           {
                              var_index[0]++;
                              off_proc_flag[n]= 
                                 hypre_CollapseStencilToStencil(Aee,
                                                                fgrid_edge,
                                                                part,
                                                                t,
                                                                var_index,
                                                                1,
                                                                0,
                                                                &stencil_vals);
                              /* put extracted stencil_vals into tridiagonal matrix */
                              lower[n-1]= stencil_vals[0];
                              diag[n-1] = stencil_vals[1];
                              upper[n-1]= stencil_vals[2];
                              hypre_TFree(stencil_vals);
                           }

                           /* solve systems to get weights. Must adjust face_w's so
                              that the stencil entry contributes. */
                           face_w1[0]*= -lower[0];
                           face_w2[rfactor[0]-2]*= -upper[rfactor[0]-2];
                           hypre_TriDiagSolve(diag, upper, lower, face_w1, rfactor[0]-1);
                           hypre_TriDiagSolve(diag, upper, lower, face_w2, rfactor[0]-1);

                           /* place weights into vals_edgeEdge */
                           for (n= 1; n< rfactor[0]; n++)
                           {
                              if (!off_proc_flag[n])  /* off_proc_flag= 1 if offproc */
                              {
                                 jedge_Edge[k]= rank;
                                 vals_edgeEdge[k]= face_w1[n-1]; /* lower end connection */
                                 k++;

                                 jedge_Edge[k]= rank2;
                                 vals_edgeEdge[k]= face_w2[n-1]; /* upper end connection */
                                 k++;
                              }
                           }
                        }  /* for (p= 0; p< rfactor[2]; p++) */

                     }
                     hypre_BoxLoop1End(m);
                  }  /* hypre_ForBoxI(i, fboxes) */
                  break;
               }

            }  /* switch */
         }     /* for (t= 0; t< Edge_nvars; t++) */

         hypre_TFree(boxoffset);
      }  /* for (part= 0; part< nparts; part++) */

      hypre_TFree(upper);
      hypre_TFree(lower);
      hypre_TFree(diag);
      hypre_TFree(face_w1);
      hypre_TFree(face_w2);
      hypre_TFree(off_proc_flag);
   }  /* if (ndim == 3) */

   /* generate the interior interpolation weights/info */
   for (part= 0; part< nparts; part++)
   {
      p_fgrid= hypre_SStructGridPGrid(fgrid_edge, part);  /* edge grid */
      Edge_nvars= hypre_SStructPGridNVars(p_fgrid);
      Edge_vartypes= hypre_SStructPGridVarTypes(p_fgrid);
      p_cgrid= hypre_SStructGridPGrid(cgrid_edge, part);  /* Edge grid */

      /* note that fboxes are the contracted CELL boxes. Will get the correct
         variable grid extents. */
      fboxes= contract_fedgeBoxes[part];

      for (t= 0; t< Edge_nvars; t++)
      {
         var      = Edge_vartypes[t];
         var_fgrid=  hypre_SStructPGridVTSGrid(p_fgrid, var);
         box_array= hypre_StructGridBoxes(var_fgrid);

         switch(var)
         {
            case 2:
            {
               /* 2-d x_face = x_edge, can be interior or on X_Edge */
               hypre_ForBoxI(i, fboxes)
               {
                  cellbox= hypre_BoxArrayBox(fboxes, i);
                  vbox   = hypre_BoxArrayBox(box_array, i);
                  hypre_CopyIndex(Edge_cstarts[part][i], cstart);

                  /* adjust the project cellbox to the variable box */
                  hypre_CopyBox(cellbox, &copy_box);
                  hypre_SubtractIndex(hypre_BoxIMin(&copy_box), varoffsets[var],
                                      hypre_BoxIMin(&copy_box));

                  hypre_BoxGetSize(&copy_box, loop_size);
                  hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                              loop_size);
                  hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                  hypre_BoxLoop1Begin(loop_size,
                                      &copy_box, start, rfactor, r);
#if 0
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,r,findex,p,n,cindex,entry,rank,var_index,k
#include "hypre_box_smp_forloop.h"
#else
                  hypre_BoxLoopSetOneBlock();
#endif
                  hypre_BoxLoop1For(loopi, loopj, loopk, r)
                  {
                     hypre_SetIndex(findex, loopi, loopj, loopk);

                     /*****************************************************
                      * Where the fine edge lies wrt the coarse edge:
                      * Since we stride by rfactor, [loopi,loopj,loopk] is
                      * the coarse index. No coarsening needed, i.e.,
                      * cindex= findex. 
                      *
                      * Loop over the interior fine edges in an agglomerate. 
                      *****************************************************/
                     for (p= 1; p< rfactor[0]; p++)
                     {
                        for (n= 0; n< rfactor[1]; n++)
                        {
                           hypre_CopyIndex(findex, cindex);
                           hypre_AddIndex(cindex, cstart, cindex);

                           /*interior of Face. Extract the four coarse Edge
                             (x_Edge ijk & (i-1,j,k) and y_Edge ijk & (i,j-1,k)
                             column ranks. No weights determined. */
                           hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex,
                                                            t, &entry);
                           hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                                 matrix_type);
                           jedge_Edge[k]= rank;
                           k++;

                           hypre_SubtractIndex(cindex, ishift, var_index);
                           hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                            t, &entry);
                           hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                 matrix_type);
                           jedge_Edge[k]= rank;
                           k++;

                           /* y_Edges */
                           hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex,
                                                            vartype_map[3], &entry);
                           hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                                 matrix_type);
                           jedge_Edge[k]= rank;
                           k++;

                           hypre_SubtractIndex(cindex, jshift, var_index);
                           hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                            vartype_map[3], &entry);
                           hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                 matrix_type);
                           jedge_Edge[k]= rank;
                           k++;
                        }  /* for (n= 0; n< rfactor[1]; n++) */
                     }     /* for (p= 1; p< rfactor[0]; p++) */

                  }
                  hypre_BoxLoop1End(r);
               }  /* hypre_ForBoxI(i, fboxes) */

               break;
            }

            case 3:
            {
               /* 2-d y_face = y_edge, can be interior or on Y_Edge */
               hypre_ForBoxI(i, fboxes)
               {
                  cellbox= hypre_BoxArrayBox(fboxes, i);
                  vbox   = hypre_BoxArrayBox(box_array, i);
                  hypre_CopyIndex(Edge_cstarts[part][i], cstart);

                  /* adjust the project cellbox to the variable box */
                  hypre_CopyBox(cellbox, &copy_box);
                  hypre_SubtractIndex(hypre_BoxIMin(&copy_box), varoffsets[var],
                                      hypre_BoxIMin(&copy_box));

                  hypre_BoxGetSize(&copy_box, loop_size);
                  hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                              loop_size);
                  hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                  hypre_BoxLoop1Begin(loop_size,
                                      &copy_box, start, rfactor, r);
#if 0
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,r,findex,p,n,cindex,entry,rank,var_index,k
#include "hypre_box_smp_forloop.h"
#else
                  hypre_BoxLoopSetOneBlock();
#endif
                  hypre_BoxLoop1For(loopi, loopj, loopk, r)
                  {
                     hypre_SetIndex(findex, loopi, loopj, loopk);

                     /*****************************************************
                      * Where the fine edge lies wrt the coarse edge:
                      * Since we stride by rfactor, [loopi,loopj,loopk] is
                      * the coarse index. No coarsening needed, i.e.,
                      * cindex= findex.
                      *
                      * Loop over the interior fine edges in an agglomerate.
                      *****************************************************/
                     for (p= 1; p< rfactor[1]; p++)
                     {
                        for (n= 0; n< rfactor[0]; n++)
                        {
                           hypre_CopyIndex(findex, cindex);
                           hypre_AddIndex(cindex, cstart, cindex);

                           /*lies interior of Face. Extract the four coarse Edge
                             (y_Edge ijk & (i,j-1,k) and x_Edge ijk & (i-1,j,k)
                             column ranks. No weights determined. */
                           hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex,
                                                            t, &entry);
                           hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                                 matrix_type);
                           jedge_Edge[k]= rank;
                           k++;

                           hypre_SubtractIndex(cindex, jshift, var_index);
                           hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                            t, &entry);
                           hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                 matrix_type);
                           jedge_Edge[k]= rank;
                           k++;

                           /* x_Edges */
                           hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex,
                                                            vartype_map[2], &entry);
                           hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                                 matrix_type);
                           jedge_Edge[k]= rank;
                           k++;

                           hypre_SubtractIndex(cindex, ishift, var_index);
                           hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                            vartype_map[2], &entry);
                           hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                 matrix_type);
                           jedge_Edge[k]= rank;
                           k++;
                        }  /* for (n= 0; n< rfactor[0]; n++) */
                     }     /* for (p= 1; p< rfactor[1]; p++) */

                  }
                  hypre_BoxLoop1End(r);
               }  /* hypre_ForBoxI(i, fboxes) */

               break;
            }

            case 5:
            {
               /* 3-d x_edge, must be interior */
               hypre_ForBoxI(i, fboxes)
               {
                  cellbox= hypre_BoxArrayBox(fboxes, i);
                  vbox   = hypre_BoxArrayBox(box_array, i);
                  hypre_CopyIndex(Edge_cstarts[part][i], cstart);

                  /* adjust the project cellbox to the variable box */
                  hypre_CopyBox(cellbox, &copy_box);
                  hypre_SubtractIndex(hypre_BoxIMin(&copy_box), varoffsets[var],
                                      hypre_BoxIMin(&copy_box));

                  hypre_BoxGetSize(&copy_box, loop_size);
                  hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                              loop_size);
                  hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                  hypre_BoxLoop1Begin(loop_size,
                                      &copy_box, start, rfactor, r);
#if 0
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,r,findex,p,n,m,cindex,entry,rank,var_index,k
#include "hypre_box_smp_forloop.h"
#else
                  hypre_BoxLoopSetOneBlock();
#endif
                  hypre_BoxLoop1For(loopi, loopj, loopk, r)
                  {
                     hypre_SetIndex(findex, loopi, loopj, loopk);

                     /*****************************************************
                      * Where the fine edge lies wrt the coarse edge:
                      * Since we stride by rfactor, [loopi,loopj,loopk] is
                      * the coarse index. No coarsening needed, i.e.,
                      * cindex= findex.
                      *
                      * Loop over the interior fine edges in an agglomerate.
                      *****************************************************/
                     for (p= 1; p< rfactor[2]; p++)
                     {
                        for (n= 1; n< rfactor[1]; n++)
                        {
                           for (m= 0; m< rfactor[0]; m++)
                           {
                              hypre_CopyIndex(findex, cindex);
                              hypre_AddIndex(cindex, cstart, cindex);

                              /***********************************************
                               * Interior.
                               * x_Edge ijk, (i,j-1,k), (i,j-1,k-1), (i,j,k-1)
                               * y_Edge ijk, (i-1,j,k), (i-1,j,k-1), (i,j,k-1)
                               * z_Edge ijk, (i-1,j,k), (i-1,j-1,k), (i,j-1,k)
                               *
                               * vals_edgeEdge's are not set.
                               ***********************************************/
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                                    matrix_type);
                              jedge_Edge[k]= rank;
                              k++;

                              hypre_SubtractIndex(cindex, jshift, var_index);
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k]= rank;
                              k++;

                              hypre_SubtractIndex(var_index, kshift, var_index);
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k]= rank;
                              k++;

                              hypre_AddIndex(var_index, jshift, var_index);
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k]= rank;
                              k++;

                              /* y_Edge */
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex,
                                                               vartype_map[6], &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                                    matrix_type);
                              jedge_Edge[k]= rank;
                              k++;

                              hypre_SubtractIndex(cindex, ishift, var_index);
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               vartype_map[6], &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k]= rank;
                              k++;

                              hypre_SubtractIndex(var_index, kshift, var_index);
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               vartype_map[6], &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k]= rank;
                              k++;

                              hypre_AddIndex(var_index, ishift, var_index);
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               vartype_map[6], &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k]= rank;
                              k++;

                              /* z_Edge */
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex,
                                                               vartype_map[7], &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                                    matrix_type);
                              jedge_Edge[k]= rank;
                              k++;

                              hypre_SubtractIndex(cindex, ishift, var_index);
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               vartype_map[7], &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k]= rank;
                              k++;

                              hypre_SubtractIndex(var_index, jshift, var_index);
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               vartype_map[7], &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k]= rank;
                              k++;

                              hypre_AddIndex(var_index, ishift, var_index);
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               vartype_map[7], &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k]= rank;
                              k++;
                           }  /* for (m= 0; m< rfactor[0]; m++) */
                        }     /* for (n= 1; n< rfactor[1]; n++) */
                     }        /* for (p= 1; p< rfactor[2]; p++) */
                  }
                  hypre_BoxLoop1End(r);
               }  /* hypre_ForBoxI(i, fboxes) */

               break;
            }

            case 6:
            {
               /* 3-d y_edge, must be interior */
               hypre_ForBoxI(i, fboxes)
               {
                  cellbox= hypre_BoxArrayBox(fboxes, i);
                  vbox   = hypre_BoxArrayBox(box_array, i);
                  hypre_CopyIndex(Edge_cstarts[part][i], cstart);

                  /* adjust the project cellbox to the variable box */
                  hypre_CopyBox(cellbox, &copy_box);
                  hypre_SubtractIndex(hypre_BoxIMin(&copy_box), varoffsets[var],
                                      hypre_BoxIMin(&copy_box));

                  hypre_BoxGetSize(&copy_box, loop_size);
                  hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                              loop_size);
                  hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                  hypre_BoxLoop1Begin(loop_size,
                                      &copy_box, start, rfactor, r);
#if 0
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,r,findex,p,n,m,cindex,entry,rank,var_index,k
#include "hypre_box_smp_forloop.h"
#else
                  hypre_BoxLoopSetOneBlock();
#endif
                  hypre_BoxLoop1For(loopi, loopj, loopk, r)
                  {
                     hypre_SetIndex(findex, loopi, loopj, loopk);

                     /*****************************************************
                      * Where the fine edge lies wrt the coarse edge:
                      * Since we stride by rfactor, [loopi,loopj,loopk] is
                      * the coarse index. No coarsening needed, i.e.,
                      * cindex= findex.
                      *
                      * Loop over the interior fine edges in an agglomerate.
                      *****************************************************/
                     for (p= 1; p< rfactor[2]; p++)
                     {
                        for (n= 1; n< rfactor[0]; n++)
                        {
                           for (m= 0; m< rfactor[1]; m++)
                           {
                              hypre_CopyIndex(findex, cindex);
                              hypre_AddIndex(cindex, cstart, cindex);

                              /***********************************************
                               * Interior.
                               * y_Edge ijk, (i-1,j,k), (i-1,j,k-1), (i,j,k-1)
                               * z_Edge ijk, (i-1,j,k), (i-1,j-1,k), (i,j-1,k)
                               * x_Edge ijk, (i,j-1,k), (i,j-1,k-1), (i,j,k-1)
                               *
                               * vals_edgeEdge's are not set.
                               ***********************************************/
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                                    matrix_type);
                              jedge_Edge[k]= rank;
                              k++;

                              hypre_SubtractIndex(cindex, ishift, var_index);
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k]= rank;
                              k++;

                              hypre_SubtractIndex(var_index, kshift, var_index);
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k]= rank;
                              k++;

                              hypre_AddIndex(var_index, ishift, var_index);
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k]= rank;
                              k++;

                              /* z_Edge */
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex,
                                                               vartype_map[7], &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                                    matrix_type);
                              jedge_Edge[k]= rank;
                              k++;

                              hypre_SubtractIndex(cindex, ishift, var_index);
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               vartype_map[7], &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k]= rank;
                              k++;

                              hypre_SubtractIndex(var_index, jshift, var_index);
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               vartype_map[7], &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k]= rank;
                              k++;

                              hypre_AddIndex(var_index, ishift, var_index);
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               vartype_map[7], &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k]= rank;
                              k++;

                              /* x_Edge */
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex,
                                                               vartype_map[5], &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                                    matrix_type);
                              jedge_Edge[k]= rank;
                              k++;

                              hypre_SubtractIndex(cindex, jshift, var_index);
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               vartype_map[5], &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k]= rank;
                              k++;

                              hypre_SubtractIndex(var_index, kshift, var_index);
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               vartype_map[5], &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k]= rank;
                              k++;

                              hypre_AddIndex(var_index, jshift, var_index);
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               vartype_map[5], &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k]= rank;
                              k++;
                           }  /* for (m= 0; m< rfactor[1]; m++) */
                        }     /* for (n= 1; n< rfactor[0]; n++) */
                     }        /* for (p= 1; p< rfactor[2]; p++) */

                  }
                  hypre_BoxLoop1End(r);
               }  /* hypre_ForBoxI(i, fboxes) */

               break;
            }

            case 7:
            {
               /* 3-d z_edge, only the interior */
               hypre_ForBoxI(i, fboxes)
               {
                  cellbox= hypre_BoxArrayBox(fboxes, i);
                  vbox   = hypre_BoxArrayBox(box_array, i);
                  hypre_CopyIndex(Edge_cstarts[part][i], cstart);

                  /* adjust the project cellbox to the variable box */
                  hypre_CopyBox(cellbox, &copy_box);
                  hypre_SubtractIndex(hypre_BoxIMin(&copy_box), varoffsets[var],
                                      hypre_BoxIMin(&copy_box));

                  hypre_BoxGetSize(&copy_box, loop_size);
                  hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                              loop_size);
                  hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                  hypre_BoxLoop1Begin(loop_size,
                                      &copy_box, start, rfactor, r);
#if 0
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,r,findex,p,n,m,cindex,entry,rank,var_index,k
#include "hypre_box_smp_forloop.h"
#else
                  hypre_BoxLoopSetOneBlock();
#endif
                  hypre_BoxLoop1For(loopi, loopj, loopk, r)
                  {
                     hypre_SetIndex(findex, loopi, loopj, loopk);

                     /*****************************************************
                      * Where the fine edge lies wrt the coarse edge:
                      * Since we stride by rfactor, [loopi,loopj,loopk] is
                      * the coarse index. No coarsening needed, i.e.,
                      * cindex= findex.
                      *
                      * Loop over the interior fine edges in an agglomerate.
                      *****************************************************/
                     for (p= 1; p< rfactor[1]; p++)
                     {
                        for (n= 1; n< rfactor[0]; n++)
                        {
                           for (m= 0; m< rfactor[2]; m++)
                           {
                              hypre_CopyIndex(findex, cindex);
                              hypre_AddIndex(cindex, cstart, cindex);

                              /*************************************************
                               * Interior.
                               * z_Edge ijk, (i-1,j,k), (i-1,j-1,k), (i,j-1,k)
                               * x_Edge ijk, (i,j-1,k), (i,j-1,k-1), (i,j,k-1)
                               * y_Edge ijk, (i-1,j,k), (i-1,j,k-1), (i,j,k-1)
                               *
                               * vals_edgeEdge's are not set.
                               *************************************************/
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                                    matrix_type);
                              jedge_Edge[k]= rank;
                              k++;

                              hypre_SubtractIndex(cindex, ishift, var_index);
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k]= rank;
                              k++;

                              hypre_SubtractIndex(var_index, jshift, var_index);
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k]= rank;
                              k++;

                              hypre_AddIndex(var_index, ishift, var_index);
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k]= rank;
                              k++;

                              /* x_Edge */
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex,
                                                               vartype_map[5], &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                                    matrix_type);
                              jedge_Edge[k]= rank;
                              k++;

                              hypre_SubtractIndex(cindex, jshift, var_index);
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               vartype_map[5], &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k]= rank;
                              k++;

                              hypre_SubtractIndex(var_index, kshift, var_index);
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               vartype_map[5], &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k]= rank;
                              k++;

                              hypre_AddIndex(var_index, jshift, var_index);
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               vartype_map[5], &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k]= rank;
                              k++;

                              /* y_Edge */
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex,
                                                               vartype_map[6], &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                                    matrix_type);
                              jedge_Edge[k]= rank;
                              k++;

                              hypre_SubtractIndex(cindex, ishift, var_index);
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               vartype_map[6], &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k]= rank;
                              k++;

                              hypre_SubtractIndex(var_index, kshift, var_index);
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               vartype_map[6], &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k]= rank;
                              k++;

                              hypre_AddIndex(var_index, ishift, var_index);
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               vartype_map[6], &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k]= rank;
                              k++;
                           }  /* for (m= 0; m< rfactor[2]; m++) */
                        }     /* for (n= 1; n< rfactor[0]; n++) */
                     }        /* for (p= 1; p< rfactor[1]; p++) */

                  }
                  hypre_BoxLoop1End(r);
               }  /* hypre_ForBoxI(i, fboxes) */

               break;
            }

         }  /* switch */
      }     /* for (t= 0; t< Edge_nvars; t++) */
   }        /* for (part= 0; part< nparts; part++) */
   hypre_TFree(bdryedge_location);

   HYPRE_IJMatrixSetValues(edge_Edge, size1, ncols_edgeEdge,
                           (const HYPRE_Int*) iedgeEdge, (const HYPRE_Int*) jedge_Edge,
                           (const double*) vals_edgeEdge);
   HYPRE_IJMatrixAssemble((HYPRE_IJMatrix) edge_Edge);

   hypre_TFree(ncols_edgeEdge);
   hypre_TFree(iedgeEdge);
   hypre_TFree(jedge_Edge);
   hypre_TFree(vals_edgeEdge);

   /* n_CtoVbox[part][cellboxi][var]  & CtoVboxnums[part][cellboxi][var][nvboxes] */
   for (part= 0; part< nparts; part++)
   {
      p_fgrid= hypre_SStructGridPGrid(fgrid_edge, part);
      Edge_nvars= hypre_SStructPGridNVars(p_fgrid);

      var_fgrid= hypre_SStructPGridCellSGrid(p_fgrid);
      fboxes   = hypre_StructGridBoxes(var_fgrid);
      hypre_ForBoxI(j, fboxes)
      {
         for (t= 0; t< Edge_nvars; t++)
         {
            hypre_TFree(CtoVboxnums[part][j][t]);
         }
         hypre_TFree(n_CtoVbox[part][j]);
         hypre_TFree(CtoVboxnums[part][j]);
      }
      hypre_TFree(n_CtoVbox[part]);
      hypre_TFree(CtoVboxnums[part]);
   }
   hypre_TFree(n_CtoVbox);
   hypre_TFree(CtoVboxnums);

   for (part= 0; part< nparts; part++)
   {
      p_fgrid= hypre_SStructGridPGrid(fgrid_edge, part);
      var_fgrid= hypre_SStructPGridCellSGrid(p_fgrid);
      fboxes   = hypre_StructGridBoxes(var_fgrid);

      hypre_BoxArrayDestroy(contract_fedgeBoxes[part]);
      hypre_TFree(Edge_cstarts[part]);
      hypre_TFree(upper_shifts[part]);
      hypre_TFree(lower_shifts[part]);
      hypre_TFree(cfbox_mapping[part]);
      hypre_TFree(fcbox_mapping[part]);
      hypre_TFree(fupper_ranks[part]);
      hypre_TFree(flower_ranks[part]);
      hypre_TFree(cupper_ranks[part]);
      hypre_TFree(clower_ranks[part]);
   }

   hypre_TFree(contract_fedgeBoxes);
   hypre_TFree(Edge_cstarts);
   hypre_TFree(upper_shifts);
   hypre_TFree(lower_shifts);
   hypre_TFree(cfbox_mapping);
   hypre_TFree(fcbox_mapping);
   hypre_TFree(fupper_ranks);
   hypre_TFree(flower_ranks);
   hypre_TFree(cupper_ranks);
   hypre_TFree(clower_ranks);

   hypre_TFree(varoffsets);
   hypre_TFree(vartype_map);

   if (ndim > 2)
   {
      (PTopology ->  Face_iedge)   = Face_iedge;
      (PTopology ->  Element_Face) = Element_Face;
   }
   (PTopology ->  Element_iedge)= Element_iedge;
   (PTopology ->  Edge_iedge)   = Edge_iedge;
   (PTopology ->  Element_Edge) = Element_Edge;

   return edge_Edge;
}

/*--------------------------------------------------------------------------
 * hypre_CollapseStencilToStencil: Collapses 3d stencil shape & values to 
 * a 2d 3-point stencil: collapsed_vals= [ldiag diag udiag].
 * Algo:
 *    1) Given the collapsing direction & the collapsed stencil pattern,
 *       group the ranks into three collapsed sets: diag_ranks, ldiag_ranks, 
 *       udiag_ranks.
 *    2) concatenate these sets, marking the set location
 *    3) qsort the concatenated set and the col_inds 
 *    4) search compare the two sorted arrays to compute the collapsed vals.
 *
 *  Example, suppose collapsing to y_edges. Then the new_stencil pattern
 *    is [n c s]^t and we need to collapse in the x direction to get this
 *    3-pt stencil: collapse_dir= 0 & new_stencil_dir= 1.
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_CollapseStencilToStencil(hypre_ParCSRMatrix     *Aee,
                               hypre_SStructGrid      *grid,
                               HYPRE_Int               part,
                               HYPRE_Int               var,
                               hypre_Index             pt_location,
                               HYPRE_Int               collapse_dir,
                               HYPRE_Int               new_stencil_dir,
                               double                **collapsed_vals_ptr)
{
   HYPRE_Int                ierr= 0;

   HYPRE_Int                matrix_type= HYPRE_PARCSR;
   HYPRE_Int                start_rank = hypre_ParCSRMatrixFirstRowIndex(Aee);
   HYPRE_Int                end_rank   = hypre_ParCSRMatrixLastRowIndex(Aee);

   hypre_BoxManEntry       *entry;

   HYPRE_Int               *ranks;
   HYPRE_Int               *marker;     /* marker to record the rank groups */
   HYPRE_Int                max_ranksize= 9;

   double                  *collapsed_vals;

   hypre_Index              index1, index2, zero_index;

   HYPRE_Int                size, *col_inds, *col_inds2;
   double                  *values;
   HYPRE_Int                rank, row_rank, *swap_inds;

   HYPRE_Int                i, j, m, centre, found;
   HYPRE_Int                getrow_ierr;
   HYPRE_Int                cnt;

   hypre_SetIndex(zero_index,0,0,0);

   /* create the collapsed stencil coefficients. Three components. */
   collapsed_vals= hypre_CTAlloc(double, 3); 

   /* check if the row corresponding to pt_location is on this proc. If
      not, return an identity row. THIS SHOULD BE CORRECTED IN THE FUTURE
      TO GIVE SOMETHING MORE REASONABLE. */
   hypre_SStructGridFindBoxManEntry(grid, part, pt_location, var, &entry);
   hypre_SStructBoxManEntryGetGlobalRank(entry, pt_location, &rank, matrix_type);
   if (rank < start_rank || rank > end_rank)
   {
      collapsed_vals[1] = 1.0;
      *collapsed_vals_ptr= collapsed_vals;
      ierr= 1;
      return ierr;
   }

   /* Extract the ranks of the collapsed stencil pattern. Since only like-var
      collapsing, we assume that max stencil size is 9. This agrees with the
      assumed pattern surrounding pt_location. Concatenating done. */
   ranks = hypre_TAlloc(HYPRE_Int, max_ranksize);
   marker= hypre_TAlloc(HYPRE_Int, max_ranksize);

   cnt= 0;
   for (j= -1; j<= 1; j++)
   {
      hypre_CopyIndex(pt_location, index1);
      index1[new_stencil_dir]+= j;

      for (i= -1; i<= 1; i++)
      {
         hypre_CopyIndex(index1, index2);
         index2[collapse_dir]+= i;
          
         hypre_SStructGridFindBoxManEntry(grid, part, index2, var, &entry);
         if (entry)
         {
            hypre_SStructBoxManEntryGetGlobalRank(entry, index2, &rank, matrix_type);
            ranks[cnt] = rank;
            marker[cnt]= j+1;

            /* mark centre component- entry!=NULL always */
            if ( (!i) && (!j) )
            {
               centre= cnt;
            }
            cnt++;
         }
      }
   }

   /* Grab the row corresponding to index pt_location. rank located in location
      centre of ranks, i.e., rank for index2= pt_location. Mark location of values,
      which will record the original location of values after the sorting. */
   row_rank= ranks[centre];
   getrow_ierr= HYPRE_ParCSRMatrixGetRow((HYPRE_ParCSRMatrix) Aee, row_rank, 
                                         &size, &col_inds, &values);
   if (getrow_ierr < 0)
      hypre_printf("offproc collapsing problem");

   swap_inds= hypre_TAlloc(HYPRE_Int, size);
   col_inds2= hypre_TAlloc(HYPRE_Int, size);
   for (i= 0; i< size; i++)
   {
      swap_inds[i]= i;
      col_inds2[i]= col_inds[i];
   }

   /* qsort ranks & col_inds */
   hypre_qsort2i(ranks, marker, 0, cnt-1);
   hypre_qsort2i(col_inds2, swap_inds, 0, size-1);

   /* search for values to collapse */
   m= 0;
   for (i= 0; i< cnt; i++)
   {
      found= 0;
      while (!found)
      {
         if (ranks[i] != col_inds2[m])
         {
            m++;
         }
         else
         {
            collapsed_vals[marker[i]]+= values[swap_inds[m]];
            m++;
            break; /* break out of while loop */
         }
      }  /* while (!found) */
   }  /* for (i= 0; i< cnt; i++) */

   HYPRE_ParCSRMatrixRestoreRow((HYPRE_ParCSRMatrix) Aee, row_rank, &size, 
                                &col_inds, &values);

   hypre_TFree(col_inds2);
   hypre_TFree(ranks);
   hypre_TFree(marker);
   hypre_TFree(swap_inds);

   *collapsed_vals_ptr= collapsed_vals;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_TriDiagSolve: Direct tridiagonal solve
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_TriDiagSolve(double     *diag,
                   double     *upper,
                   double     *lower,
                   double     *rhs,
                   HYPRE_Int   size)
{
   HYPRE_Int       ierr= 0;

   HYPRE_Int       i, size1;
   double         *copy_diag; 
   double          multiplier;

   size1= size-1;

   /* copy diag so that the matrix is not modified */
   copy_diag= hypre_TAlloc(double, size);
   for (i= 0; i< size; i++)
   {
      copy_diag[i]= diag[i];
   }

   /* forward substitution */
   for (i= 1; i< size; i++)
   {
      multiplier= -lower[i]/copy_diag[i-1];
      copy_diag[i]+= multiplier*upper[i-1];
      rhs[i] += multiplier*rhs[i-1];
   }

   /* backward substitution */
   rhs[size1]/= copy_diag[size1];
   for (i= size1-1; i>= 0; i--)
   {
      rhs[i]= (rhs[i] - upper[i]*rhs[i+1])/copy_diag[i];
   }

   hypre_TFree(copy_diag);

   return ierr;
}

   
