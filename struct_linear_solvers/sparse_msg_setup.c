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
#include "sparse_msg.h"

#define DEBUG 0

/*--------------------------------------------------------------------------
 * hypre_SparseMSGSetup
 *--------------------------------------------------------------------------*/

int
hypre_SparseMSGSetup( void               *SparseMSG_vdata,
                      hypre_StructMatrix *A,
                      hypre_StructVector *b,
                      hypre_StructVector *x        )
{
   hypre_SparseMSGData  *SparseMSG_data = SparseMSG_vdata;

   MPI_Comm              comm = (SparseMSG_data -> comm);
                     
   int                   relax_type = (SparseMSG_data -> relax_type);
   int                  *num_levels = (SparseMSG_data -> num_levels);
   int                   jump       = (SparseMSG_data -> jump);

   int                   max_iter;
                      
   int                   total_num_levels = 1;
   int                   num_grids = 1;

   hypre_Index           cindex;
   hypre_Index           findex;
   hypre_Index           stride;

   hypre_StructGrid    **grid_array;
   hypre_StructGrid    **P_grid_array;
                    
   hypre_StructMatrix  **A_array;
   hypre_StructMatrix  **P_array;
   hypre_StructMatrix  **RT_array;
   hypre_StructVector  **b_array;
   hypre_StructVector  **x_array;

   /* temp vectors */
   hypre_StructVector  **tx_array;
   hypre_StructVector  **r_array;
   hypre_StructVector  **e_array;

   void                **relax_data_array;
   void                **matvec_data_array;
   void                **restrict_data_array;
   void                **interp_data_array;

   hypre_StructGrid     *grid;
   hypre_BoxArray       *boxes;
   hypre_BoxArray       *all_boxes;
   int                  *processes;
   int                  *box_ranks;
   hypre_BoxArray       *base_all_boxes;
   hypre_Index           pindex;
   hypre_Index           pstride;

   /* temp Indicies */
   hypre_Index           f_pindex;
   hypre_Index           f_pstride;
   hypre_Index           tP_pindex;
   hypre_Index           tpindex;
   hypre_Index           tpstride;

   hypre_BoxArray       *P_all_boxes;
   hypre_Index           P_pindex;

   int                   num_boxes;
   int                   num_all_boxes;
   int                   dim;

   hypre_Box            *box;
   hypre_Box            *cbox;   

   int                   idmin, idmax;
   int                   i, d, l, lx, ly, lz;
   int                   index, index2, rindex;
   int                   ymin, zmin;
   int                   level, cdir;
   hypre_Index           lxyz;
                       
   int                   b_num_ghost[]  = {0, 0, 0, 0, 0, 0};
   int                   x_num_ghost[]  = {1, 1, 1, 1, 1, 1};

   int                   ierr = 0;
#if DEBUG
   char                  filename[255];
#endif


   /*-----------------------------------------------------
    * Set up coarse grids
    *-----------------------------------------------------*/

   grid           = hypre_StructMatrixGrid(A);
   boxes          = hypre_StructGridBoxes(grid);
   all_boxes      = hypre_StructGridAllBoxes(grid);
   processes      = hypre_StructGridProcesses(grid);
   box_ranks      = hypre_StructGridBoxRanks(grid);
   base_all_boxes = hypre_StructGridBaseAllBoxes(grid);
   hypre_CopyIndex(hypre_StructGridPIndex(grid),  pindex);
   hypre_CopyIndex(hypre_StructGridPStride(grid), pstride);
   num_boxes      = hypre_BoxArraySize(boxes);
   num_all_boxes  = hypre_BoxArraySize(all_boxes);
   dim            = hypre_StructGridDim(grid);

   /* store some information about the fine grid */
   hypre_CopyIndex(pindex, f_pindex);
   hypre_CopyIndex(pstride, f_pstride);

   /* compute all_boxes from base_all_boxes */
   hypre_ForBoxI(i, all_boxes)
      {
         box = hypre_BoxArrayBox(all_boxes, i);
         hypre_CopyBox(hypre_BoxArrayBox(base_all_boxes, i), box);
         hypre_ProjectBox(box, pindex, pstride);
         hypre_SparseMSGMapFineToCoarse(hypre_BoxIMin(box), pindex, pstride,
                                        hypre_BoxIMin(box));
         hypre_SparseMSGMapFineToCoarse(hypre_BoxIMax(box), pindex, pstride,
                                        hypre_BoxIMax(box));
      }

   /* allocate P_all_boxes and f_all_boxes */
   P_all_boxes = hypre_BoxArrayCreate(num_all_boxes);

   /* Compute a bounding box (cbox) used to determine
      num_levels[] and total_num_levels */
   cbox = hypre_BoxCreate();
   for (d = 0; d < dim; d++)
   {
      idmin = hypre_BoxIMinD(hypre_BoxArrayBox(all_boxes, 0), d);
      idmax = hypre_BoxIMaxD(hypre_BoxArrayBox(all_boxes, 0), d);
      for (i = 0; i < num_all_boxes; i++)
      {
         idmin = hypre_min(idmin,
                           hypre_BoxIMinD(hypre_BoxArrayBox(all_boxes, i), d));
         idmax = hypre_max(idmax,
                           hypre_BoxIMaxD(hypre_BoxArrayBox(all_boxes, i), d));
      }
      hypre_BoxIMinD(cbox, d) = idmin;
      hypre_BoxIMaxD(cbox, d) = idmax;
   }
   for (d = dim; d < 3; d++)
   {
      hypre_BoxIMinD(cbox, d) = 0;
      hypre_BoxIMaxD(cbox, d) = 0;
   }

   /* Determine num_levels[] and total_num_levels */
   for (d = 0; d < dim; d++)
   {
      idmin = hypre_BoxIMinD(cbox,d);
      idmax = hypre_BoxIMaxD(cbox,d);
      while ( idmax > idmin )
      {
         /* set cindex, findex, and stride */
         hypre_SparseMSGSetCIndex(d,cindex);
         hypre_SparseMSGSetFIndex(d,findex);
         hypre_SparseMSGSetStride(d,stride);

         /* coarsen cbox */
         hypre_ProjectBox(cbox,cindex,stride);
         hypre_SparseMSGMapFineToCoarse(hypre_BoxIMin(cbox),
                                        cindex,stride,hypre_BoxIMin(cbox));
         hypre_SparseMSGMapFineToCoarse(hypre_BoxIMax(cbox),
                                        cindex,stride,hypre_BoxIMax(cbox));
         
         /* increment level counters */
         num_levels[d]++;
         total_num_levels++;

         idmin = hypre_BoxIMinD(cbox,d);
         idmax = hypre_BoxIMaxD(cbox,d);
      }
   }

#if 0
   /* Restrict the semicoarsening to a particular direction */
   num_levels[1] = 1;
   num_levels[2] = 1;
   total_num_levels = num_levels[0];

#endif 

   /* Compute the num_grids based on num_levels[] */
   for (d = 0; d < dim; d++)
   {
      num_grids *= num_levels[d];
   }

   /* Store some variables and clean up */
   hypre_BoxDestroy(cbox);
   (SparseMSG_data->num_grids) = num_grids;
   (SparseMSG_data->total_num_levels) = total_num_levels;

   grid_array = hypre_TAlloc(hypre_StructGrid *, num_grids);
   grid_array[0] = hypre_StructGridRef(grid);
   P_grid_array = hypre_TAlloc(hypre_StructGrid *, 3*num_grids);

   for (i = 0; i < 3*num_grids; i++)
   {
      P_grid_array[i] = NULL;
   }

   /*-----------------------------------------
    * start computing coarse grids 
    -----------------------------------------*/

   if (total_num_levels > 1)
   {
      for (lz = 0; lz <= (num_levels[2]-1); lz++)
      {
         for (ly = 0; ly <= (num_levels[1]-1); ly++)
         {
            hypre_CopyIndex(f_pindex, pindex);
            hypre_CopyIndex(f_pstride, pstride); 
                        
            /* compute P_pindex, pindex, and pstride for z-dir coarsening */
            for (l = 0; l < lz; l++)
            {
               if (!l)
               {
                  hypre_SparseMSGSetCIndex(2,cindex);
                  hypre_SparseMSGSetStride(2,stride);
                  hypre_SparseMSGSetFIndex(2,findex);
               }
               for (d = 0; d < dim; d++)
               {
                  hypre_IndexD(P_pindex, d) = hypre_IndexD(pindex, d) +
                     hypre_IndexD(findex, d) * hypre_IndexD(pstride, d);
                  hypre_IndexD(pindex, d) +=
                     hypre_IndexD(cindex, d) * hypre_IndexD(pstride, d);
                  hypre_IndexD(pstride, d) *= hypre_IndexD(stride, d);
               }               
            }

            /* compute P_pindex, pindex, and pstride for y-dir coarsening */
            for (l = 0; l < ly; l++)
            {
               if (!l)
               {
                  hypre_SparseMSGSetCIndex(1, cindex);
                  hypre_SparseMSGSetFIndex(1, findex);
                  hypre_SparseMSGSetStride(1, stride);
               }
               for (d = 0; d < dim; d++)
               {
                  hypre_IndexD(P_pindex, d) = hypre_IndexD(pindex, d) +
                     hypre_IndexD(findex, d) * hypre_IndexD(pstride, d);
                  hypre_IndexD(pindex, d) +=
                     hypre_IndexD(cindex, d) * hypre_IndexD(pstride, d);
                  hypre_IndexD(pstride, d) *= hypre_IndexD(stride, d);
               }
            }
          
            /* BUILD THE COARSE GRID */
            if (ly || lz)
            {
               /* compute grid_array index */
               hypre_SparseMSGComputeArrayIndex(0, ly, lz,
                                                num_levels[0], num_levels[1],
                                                index);

               /* compute all_boxes from base_all_boxes */
               hypre_ForBoxI(i, all_boxes)
                  {
                     box = hypre_BoxArrayBox(all_boxes, i);
                     hypre_CopyBox(hypre_BoxArrayBox(base_all_boxes, i), box);
                     hypre_ProjectBox(box, pindex, pstride);
                     hypre_SparseMSGMapFineToCoarse(hypre_BoxIMin(box),
                                                    pindex, pstride,
                                                    hypre_BoxIMin(box));
                     hypre_SparseMSGMapFineToCoarse(hypre_BoxIMax(box),
                                                    pindex, pstride,
                                                    hypre_BoxIMax(box));
                  }

               /* compute local boxes */
               boxes = hypre_BoxArrayCreate(num_boxes);
               for (i = 0; i < num_boxes; i++)
               {
                  hypre_CopyBox(hypre_BoxArrayBox(all_boxes, box_ranks[i]),
                                hypre_BoxArrayBox(boxes, i));
               }
               
               grid_array[index] =
                  hypre_StructGridCreate(comm, hypre_StructGridDim(grid));
               hypre_StructGridSetBoxes(grid_array[index], boxes);
               hypre_StructGridSetGlobalInfo(grid_array[index],
                                             all_boxes, processes, box_ranks,
                                             base_all_boxes, pindex, pstride);
               hypre_StructGridAssemble(grid_array[index]);
            }
            else
            { 
               index = 0;
            }
            
            /*--------------------------------------------------
             *  semicoarsening in the x-direction 
             *-------------------------------------------------*/

            for (lx = 0; lx <= (num_levels[0]-1); lx++)
            {
               index++;
               hypre_SetIndex(lxyz, lx, ly, lz);
               
               /*-------------------------------------------------
                *  build the P interpolation grids
                *------------------------------------------------*/

               for (d = 0; d < 3; d++)
               {
                  /* set cindex, findex, and stride */
                  hypre_SparseMSGSetCIndex(d, cindex);
                  hypre_SparseMSGSetFIndex(d, findex);
                  hypre_SparseMSGSetStride(d, stride);                  
                  
                  hypre_SparseMSGComputeArrayIndex((lx + hypre_IndexX(findex)),
                                                   (ly + hypre_IndexY(findex)),
                                                   (lz + hypre_IndexZ(findex)),
                                                   num_levels[0],
                                                   num_levels[1],
                                                   index2);
                  rindex = 3*index2 + d;
                  
                  if (hypre_IndexD(lxyz,d) < (num_levels[d]-1))
                  {
                     /* copy P_pindex, pindex, pstride */
                     hypre_CopyIndex(P_pindex, tP_pindex);
                     hypre_CopyIndex(pindex, tpindex);
                     hypre_CopyIndex(pstride, tpstride);

                     /* compute new tP_pindex, tpindex, and tpstride */
                     for (i = 0; i < dim; i++)
                     {
                        hypre_IndexD(tP_pindex, i) = hypre_IndexD(tpindex, i) +
                           hypre_IndexD(findex, i) * hypre_IndexD(tpstride, i);
                        hypre_IndexD(tpindex, i) +=
                           hypre_IndexD(cindex, i) * hypre_IndexD(tpstride, i);
                        hypre_IndexD(tpstride, i) *= hypre_IndexD(stride, i);
                     }   
               
                     /* build from all_boxes (reduces communication) */
                     for (i = 0; i < num_all_boxes; i++)
                     {
                        hypre_CopyBox(hypre_BoxArrayBox(all_boxes, i),
                                      hypre_BoxArrayBox(P_all_boxes, i));
                     }
                     hypre_ProjectBoxArray(P_all_boxes, findex, stride);
                     for (i = 0; i < num_all_boxes; i++)
                     {
                        box = hypre_BoxArrayBox(P_all_boxes, i);
                        hypre_SparseMSGMapFineToCoarse(hypre_BoxIMin(box),
                                                       findex, stride,
                                                       hypre_BoxIMin(box));
                        hypre_SparseMSGMapFineToCoarse(hypre_BoxIMax(box),
                                                       findex, stride,
                                                       hypre_BoxIMax(box));
                     }
                     
                     /* compute local boxes */
                     boxes = hypre_BoxArrayCreate(num_boxes);
                     for (i = 0; i < num_boxes; i++)
                     {
                        hypre_CopyBox(hypre_BoxArrayBox(P_all_boxes,
                                                        box_ranks[i]),
                                      hypre_BoxArrayBox(boxes, i));
                     }
                     
                     P_grid_array[rindex] =
                        hypre_StructGridCreate(comm, hypre_StructGridDim(grid));
                     hypre_StructGridSetBoxes(P_grid_array[rindex], boxes);
                     hypre_StructGridSetGlobalInfo(P_grid_array[rindex],
                                                   P_all_boxes, processes,
                                                   box_ranks, base_all_boxes,
                                                   tP_pindex, tpstride);
                     hypre_StructGridAssemble(P_grid_array[rindex]);      
                  }
               }

               if (lx < (num_levels[0]-1))
               {
                  /* set cindex, findex, and stride */
                  hypre_SparseMSGSetCIndex(0, cindex);
                  hypre_SparseMSGSetFIndex(0, findex);
                  hypre_SparseMSGSetStride(0, stride);   
                  
                  /* compute new P_pindex, pindex, and pstride */
                  for (d = 0; d < dim; d++)
                  {
                     hypre_IndexD(P_pindex, d) = hypre_IndexD(pindex, d) +
                        hypre_IndexD(findex, d) * hypre_IndexD(pstride, d);
                     hypre_IndexD(pindex, d) +=
                        hypre_IndexD(cindex, d) * hypre_IndexD(pstride, d);
                     hypre_IndexD(pstride, d) *= hypre_IndexD(stride, d);
                  }   
                  
                  /*---------------------------------------
                   * build the coarse grid
                   *---------------------------------------*/
                  
                  /* coarsen the grid by coarsening all_boxes
                     (reduces communication) */
                  hypre_ProjectBoxArray(all_boxes, cindex, stride);
                  for (i = 0; i < num_all_boxes; i++)
                  {
                     box = hypre_BoxArrayBox(all_boxes, i);
                     hypre_SparseMSGMapFineToCoarse(hypre_BoxIMin(box),
                                                    cindex, stride,
                                                    hypre_BoxIMin(box));
                     hypre_SparseMSGMapFineToCoarse(hypre_BoxIMax(box),
                                                    cindex, stride,
                                                    hypre_BoxIMax(box));
                  }
                  
                  /* compute local boxes */
                  boxes = hypre_BoxArrayCreate(num_boxes);
                  for (i = 0; i < num_boxes; i++)
                  {
                     hypre_CopyBox(hypre_BoxArrayBox(all_boxes, box_ranks[i]),
                                   hypre_BoxArrayBox(boxes, i));
                  }
                  
                  grid_array[index] =
                     hypre_StructGridCreate(comm, hypre_StructGridDim(grid));
                  hypre_StructGridSetBoxes(grid_array[index], boxes);
                  hypre_StructGridSetGlobalInfo(grid_array[index],
                                                all_boxes, processes,
                                                box_ranks, base_all_boxes,
                                                pindex, pstride);
                  hypre_StructGridAssemble(grid_array[index]);
               }
            }
         }
      }
   }

   /* free up some things */
   hypre_BoxArrayDestroy(P_all_boxes);

   (SparseMSG_data -> grid_array)     = grid_array;
   (SparseMSG_data -> P_grid_array)   = P_grid_array;


   /*-----------------------------------------------------
    * Set up matrix and vector structures
    *-----------------------------------------------------*/

   A_array  = hypre_TAlloc(hypre_StructMatrix *, num_grids);
   P_array  = hypre_TAlloc(hypre_StructMatrix *, 3*num_grids);
   RT_array = hypre_TAlloc(hypre_StructMatrix *, 3*num_grids);
   b_array  = hypre_TAlloc(hypre_StructVector *, num_grids);
   x_array  = hypre_TAlloc(hypre_StructVector *, num_grids);
   tx_array = hypre_TAlloc(hypre_StructVector *, num_grids);
   r_array  = hypre_TAlloc(hypre_StructVector *, num_grids);
   e_array  = r_array;

   A_array[0] = hypre_StructMatrixRef(A);
   b_array[0] = hypre_StructVectorRef(b);
   x_array[0] = hypre_StructVectorRef(x);

   tx_array[0] = hypre_StructVectorCreate(comm, grid_array[0]);
   hypre_StructVectorSetNumGhost(tx_array[0], x_num_ghost);
   hypre_StructVectorInitialize(tx_array[0]);

   r_array[0] = hypre_StructVectorCreate(comm, grid_array[0]);
   hypre_StructVectorSetNumGhost(r_array[0], x_num_ghost);
   hypre_StructVectorInitialize(r_array[0]);

   for (i = 0; i < 3*num_grids; i++)
   {
      P_array[i] = NULL;
      RT_array[i] = NULL;
   }

   /*------------------------------------------------------
    *  Set up multigrid operators
    *-----------------------------------------------------*/

   relax_data_array    = hypre_TAlloc(void *, num_grids);
   matvec_data_array   = hypre_TAlloc(void *, num_grids);
   restrict_data_array = hypre_TAlloc(void *, 3*num_grids);
   interp_data_array   = hypre_TAlloc(void *, 3*num_grids);

   /*------------------------------------------------------
    *  Initialize structures
    *-----------------------------------------------------*/

   if (total_num_levels > 1)
   {
      for (lz = 0; lz <= (num_levels[2]-1); lz++)
      {
         for (ly = 0; ly <= (num_levels[1]-1); ly++)
         {
            /* compute the array index */
            lx = 0;
            level = lx + ly + lz;
            hypre_SparseMSGComputeArrayIndex(lx,ly,lz,
                                             num_levels[0], num_levels[1],
                                             index);
            
            /* setup hypre_Index lxyz */
            hypre_SetIndex(lxyz, lx, ly, lz);
            
            for (d = 0; d < dim; d++)
            {
               rindex = 3*index + d;
               hypre_SparseMSGSetCIndex(d, cindex);
               hypre_SparseMSGSetFIndex(d, findex);
               hypre_SparseMSGSetStride(d, stride);
               
               /* initialize and setup RT, A, P, x, b, tx, and r (d-dir) */
               if (hypre_IndexD(lxyz, d) < (num_levels[d]-1))
               {
                  hypre_SparseMSGComputeArrayIndex(hypre_IndexX(findex),
                                                   (ly + hypre_IndexY(findex)),
                                                   (lz + hypre_IndexZ(findex)),
                                                   num_levels[0],
                                                   num_levels[1],
                                                   index2);
                  
                  /* set up interpolation operator */
                  P_array[rindex] =
                     hypre_PFMGCreateInterpOp(A_array[index],
                                              P_grid_array[(3*index2)+d], d);
                  hypre_StructMatrixInitialize(P_array[rindex]);
                  
                  /* set up interpolation either from A or from a previous P */
                  if((!lx + !ly + !lz - !hypre_IndexD(lxyz,d))==2)
                  {
                     hypre_PFMGSetupInterpOp(A_array[index], d,
                                             findex, stride, 
                                             P_array[rindex]);
                  }
                  else
                  {
                     hypre_Index txyz;
                     hypre_Index tstride;
                     int         tindex;
                     hypre_ClearIndex(txyz);
                     hypre_IndexD(txyz, d) = hypre_IndexD(lxyz, d);
                     hypre_SparseMSGComputeArrayIndex(hypre_IndexX(txyz),
                                                      hypre_IndexY(txyz),
                                                      hypre_IndexZ(txyz),
                                                      num_levels[0],
                                                      num_levels[1],
                                                      tindex);
                     tindex = 3*tindex + d;
                     hypre_SetIndex(tstride,
                                    (int)pow(2, (lx - hypre_IndexD(txyz, 0))),
                                    (int)pow(2, (ly - hypre_IndexD(txyz, 1))),
                                    (int)pow(2, (lz - hypre_IndexD(txyz, 2))));
                     hypre_SparseMSGSetupInterpOp(P_array[tindex],
                                                  cindex, tstride,
                                                  P_array[rindex]);
                  }

                  /* set up restriction */
                  if (hypre_StructMatrixSymmetric(A))
                  {
                     RT_array[rindex] = P_array[rindex];
                  }
                  else 
                  {
#if 0
                     RT_array[rindex] =
                        hypre_PFMGCreateRestrictOp(A_array[index],
                                                   grid_array[index2], d);
                     hypre_StructMatrixInitialize(RT_array[rindex]);
                     hypre_PFMGSetupRestrictOp(A_array[index], tx_array[index],
                                               d, cindex, stride,
                                               RT_array[rindex]);
#endif
                  }
                  
                  /* set up coarse grid operator */
                  A_array[index2] = hypre_PFMGCreateRAPOp(RT_array[rindex],
                                                          A_array[index],
                                                          P_array[rindex], 
                                                          grid_array[index2], d);
                  hypre_StructMatrixInitialize(A_array[index2]);
                  hypre_PFMGSetupRAPOp(RT_array[rindex], A_array[index], 
                                       P_array[rindex], d, cindex, stride, 
                                       A_array[index2]);
                  
                  /* set up b, x, tx, and r_array */
                  b_array[index2] =
                     hypre_StructVectorCreate(comm, grid_array[index2]);
                  hypre_StructVectorSetNumGhost(b_array[index2],b_num_ghost);
                  hypre_StructVectorInitialize(b_array[index2]);
                  hypre_StructVectorAssemble(b_array[index2]);
                  
                  x_array[index2] =
                     hypre_StructVectorCreate(comm, grid_array[index2]);
                  hypre_StructVectorSetNumGhost(x_array[index2],x_num_ghost);
                  hypre_StructVectorInitialize(x_array[index2]);
                  hypre_StructVectorAssemble(x_array[index2]);
                  
                  tx_array[index2] =
                     hypre_StructVectorCreate(comm, grid_array[index2]);
                  hypre_StructVectorSetNumGhost(tx_array[index2],x_num_ghost);
                  hypre_StructVectorInitializeShell(tx_array[index2]);
                  hypre_StructVectorInitializeData(tx_array[index2],
                                                   hypre_StructVectorData(tx_array[0]));
                  hypre_StructVectorAssemble(tx_array[index2]);
                  
                  r_array[index2] =
                     hypre_StructVectorCreate(comm, grid_array[index2]);
                  hypre_StructVectorSetNumGhost(r_array[index2],x_num_ghost);
                  hypre_StructVectorInitialize(r_array[index2]);
                  hypre_StructVectorAssemble(r_array[index2]);                   
               }
            }
            
            /* free any coarse grid operators in the jump*/
            if ((level <= jump) && (level != 0))
            {
               hypre_StructMatrixDestroy(A_array[index]);
               A_array[index] = NULL;
            }
            
            for (lx = 1; lx <= (num_levels[0]-1); lx++)
            {
               index++;
               hypre_IndexX(lxyz)++;
               
               for (d = 0; d < dim; d++)
               {
                  rindex = 3*index + d;
                  hypre_SparseMSGSetCIndex(d, cindex);
                  hypre_SparseMSGSetFIndex(d, findex);
                  hypre_SparseMSGSetStride(d, stride);
                  
                  /* compute interpolation matrices for all directions */
                  if (hypre_IndexD(lxyz, d) < (num_levels[d]-1))
                  {
                     hypre_SparseMSGComputeArrayIndex(
                        (lx + hypre_IndexX(findex)),
                        (ly + hypre_IndexY(findex)),
                        (lz + hypre_IndexZ(findex)),
                        num_levels[0],num_levels[1], index2);       
                     
                     P_array[rindex] =
                        hypre_PFMGCreateInterpOp(A_array[index], 
                                                 P_grid_array[(3*index2)+d], d);
                     hypre_StructMatrixInitialize(P_array[rindex]);

                     /* set up interpolation from either A or previous P */
                     if((!lx + !ly + !lz - !hypre_IndexD(lxyz,d))==2)
                     {
                        hypre_PFMGSetupInterpOp(A_array[index], d,
                                                findex, stride, 
                                                P_array[rindex]);
                     }
                     else
                     {
                        hypre_Index txyz;
                        hypre_Index tstride;
                        int         tindex;
                        hypre_ClearIndex(txyz);
                        hypre_IndexD(txyz, d) = hypre_IndexD(lxyz, d);
                        hypre_SparseMSGComputeArrayIndex(
                           hypre_IndexX(txyz),
                           hypre_IndexY(txyz),
                           hypre_IndexZ(txyz),
                           num_levels[0],num_levels[1],tindex);
                        tindex = 3*tindex + d;
                        hypre_SetIndex(
                           tstride,
                           (int)pow(2, (lx - hypre_IndexD(txyz, 0))),
                           (int)pow(2, (ly - hypre_IndexD(txyz, 1))),
                           (int)pow(2, (lz - hypre_IndexD(txyz, 2))));
                        hypre_SparseMSGSetupInterpOp(
                           P_array[tindex], cindex, tstride, P_array[rindex]);
                     }
                     
                     /* set up restriction */
                     if (hypre_StructMatrixSymmetric(A))
                     {
                        RT_array[rindex] = P_array[rindex];
                     }
                     else 
                     {
#if 0
                        RT_array[rindex] =
                           hypre_PFMGCreateRestrictOp(A_array[index],
                                                      grid_array[index2],d);
                        hypre_StructMatrixInitialize(RT_array[rindex]);
                        hypre_PFMGSetupRestrictOp(
                           A_array[index], tx_array[index], 
                           d, cindex, stride, RT_array[rindex]);
#endif
                     }
                  }
               }
               
               if (lx < (num_levels[0]-1))
               {
                  level = lx + ly + lz;
                  index2 = index + 1;
                  rindex = 3*index;
                  
                  hypre_SparseMSGSetCIndex(0, cindex);
                  hypre_SparseMSGSetFIndex(0, findex);
                  hypre_SparseMSGSetStride(0, stride);

                  /* set up coarse grid operator */
                  A_array[index2] =
                     hypre_PFMGCreateRAPOp(RT_array[rindex],
                                           A_array[index],P_array[rindex],
                                           grid_array[index2],0);
                  hypre_StructMatrixInitialize(A_array[index2]);
                  hypre_PFMGSetupRAPOp(RT_array[rindex], A_array[index], 
                                       P_array[rindex], 0, cindex, stride, 
                                       A_array[index2]);
                  
                  /* set up b, x, tx, and r_array */
                  b_array[index2] =
                     hypre_StructVectorCreate(comm, grid_array[index2]);
                  hypre_StructVectorSetNumGhost(b_array[index2],b_num_ghost);
                  hypre_StructVectorInitialize(b_array[index2]);
                  hypre_StructVectorAssemble(b_array[index2]);
                  
                  x_array[index2] =
                     hypre_StructVectorCreate(comm, grid_array[index2]);
                  hypre_StructVectorSetNumGhost(x_array[index2],x_num_ghost);
                  hypre_StructVectorInitialize(x_array[index2]);
                  hypre_StructVectorAssemble(x_array[index2]);
                  
                  tx_array[index2] =
                     hypre_StructVectorCreate(comm, grid_array[index2]);
                  hypre_StructVectorSetNumGhost(tx_array[index2],x_num_ghost);
                  hypre_StructVectorInitializeShell(tx_array[index2]);
                  hypre_StructVectorInitializeData(
                     tx_array[index2], hypre_StructVectorData(tx_array[0]));
                  hypre_StructVectorAssemble(tx_array[index2]);
                  
                  r_array[index2] =
                     hypre_StructVectorCreate(comm, grid_array[index2]);
                  hypre_StructVectorSetNumGhost(r_array[index2],x_num_ghost);
                  hypre_StructVectorInitialize(r_array[index2]);
                  hypre_StructVectorAssemble(r_array[index2]);
                  
                  /* free any coarse grid operators in the jump */
                  if (level <= jump)
                  {
                     hypre_StructMatrixDestroy(A_array[index]);
                     A_array[index] = NULL;
                  }
               }
            }
         }
      }

      /* set up relax, matvec, restrict, interp_data_array */
      for (lz = 0; lz <= (num_levels[2]-1); lz++)
      {
         for (ly = 0; ly <= (num_levels[1]-1); ly++)
         {  
            for (lx = 0; lx <= (num_levels[0]-1); lx++)
            {
               level = lx + ly + lz;           
               hypre_SparseMSGComputeArrayIndex(lx,ly,lz,
                                                num_levels[0],
                                                num_levels[1],
                                                index);
               /* set hypre_Index lxyz */
               hypre_SetIndex(lxyz, lx, ly, lz);
               
               if (level > jump || level == 0)
               {
                  /* set up the residual routine */
                  matvec_data_array[index] = hypre_StructMatvecCreate();
                  hypre_StructMatvecSetup(matvec_data_array[index],
                                          A_array[index], x_array[index]);
                  relax_data_array[index] = hypre_PFMGRelaxCreate(comm);
                  hypre_PFMGRelaxSetTol(relax_data_array[index], 0.0);
                  
                  /* set up relaxation */
                  if (level < (total_num_levels-1))
                  {
                     hypre_PFMGRelaxSetType(relax_data_array[index],
                                            relax_type);
                  }
                  else
                  {
                     hypre_PFMGRelaxSetMaxIter(relax_data_array[index],1);
                     hypre_PFMGRelaxSetType(relax_data_array[index], 0);
                  }                               
                  
                  hypre_PFMGRelaxSetTempVec(relax_data_array[index], 
                                            tx_array[index]);
                  hypre_PFMGRelaxSetup(relax_data_array[index],
                                       A_array[index],  b_array[index],
                                       x_array[index]);
               }
               else
               {
                  relax_data_array[index] = NULL;
                  matvec_data_array[index] = NULL;
               }
                  
               for (d = 0; d < 3; d++)
               {
                  rindex = 3*index + d;
                  hypre_SparseMSGSetCIndex(d, cindex);
                  hypre_SparseMSGSetFIndex(d, findex);
                  hypre_SparseMSGSetStride(d, stride);
                  
                  /* compute interpolation matrices for all directions */
                  if (hypre_IndexD(lxyz, d) < (num_levels[d]-1))
                  {
                     hypre_SparseMSGComputeArrayIndex(
                        (lx + hypre_IndexX(findex)),
                        (ly + hypre_IndexY(findex)),
                        (lz + hypre_IndexZ(findex)),
                        num_levels[0], num_levels[1], index2);       
                     
                     /* set up the interpolation routine */
                     interp_data_array[rindex] = hypre_PFMGInterpCreate();
                     hypre_PFMGInterpSetup(interp_data_array[rindex],
                                           P_array[rindex],
                                           x_array[index2], e_array[index],
                                           cindex, findex, stride);
                     
                     restrict_data_array[rindex] =
                        hypre_PFMGRestrictCreate();
                     hypre_PFMGRestrictSetup(restrict_data_array[rindex],
                                             RT_array[rindex], r_array[index],
                                             b_array[index2],
                                             cindex, findex, stride);
                  }
                  else
                  {
                     interp_data_array[rindex] = NULL;
                     restrict_data_array[rindex] = NULL;
                  }
               }
            }
         }
      }
   }

   (SparseMSG_data -> A_array)  = A_array;
   (SparseMSG_data -> P_array)  = P_array;
   (SparseMSG_data -> RT_array) = RT_array;
   (SparseMSG_data -> b_array)  = b_array;
   (SparseMSG_data -> x_array)  = x_array;
   (SparseMSG_data -> tx_array) = tx_array;
   (SparseMSG_data -> r_array)  = r_array;
   (SparseMSG_data -> e_array)  = e_array;

   (SparseMSG_data -> relax_data_array)    = relax_data_array;
   (SparseMSG_data -> matvec_data_array)   = matvec_data_array;
   (SparseMSG_data -> restrict_data_array) = restrict_data_array;
   (SparseMSG_data -> interp_data_array)   = interp_data_array;

   
   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/

   if ((SparseMSG_data -> logging) > 0)
   {
      max_iter = (SparseMSG_data -> max_iter);
      (SparseMSG_data -> norms)     = hypre_TAlloc(double, max_iter);
      (SparseMSG_data -> rel_norms) = hypre_TAlloc(double, max_iter);
   }

#if DEBUG
   for (l = 0; l < num_grids; l++)
   {
      sprintf(filename, "zoutSMSG_A.%02d",l);
      hypre_StructMatrixPrint(filename, A_array[l], 0);
      for (i = 0; i < 3; i++)
      {
         if (P_array[(3*l + i)])
         {
            sprintf(filename, "zoutSMSG_P.%02d_%02d",l,i);
            hypre_StructMatrixPrint(filename, P_array[(3*l + i)], 0);
         }
      }
   }
#endif
   
   /*-----------------------------------------------------
    * Compute Interpolation and Restriction Weights 
    *-----------------------------------------------------*/

   (SparseMSG_data -> restrict_weights) = hypre_TAlloc(double, 3*num_grids);
   (SparseMSG_data -> interp_weights)   = hypre_TAlloc(double, 3*num_grids);
   hypre_SparseMSGComputeRestrictWeights(num_levels,
                                         SparseMSG_data->restrict_weights);
   hypre_SparseMSGComputeInterpWeights(num_levels,
                                       SparseMSG_data->interp_weights);

   return ierr;
}


/*--------------------------------------------------------------------------
 * Functions for computing the restriction/interpolation weights
 *--------------------------------------------------------------------------*/

int
hypre_SparseMSGComputeRestrictWeights( int    *num_levels,
                                       double *restrict_weights )
{
   int lx, ly, lz, l;
   int index;
   int restrict_count;
   int ierr = 0;

   /* Compute restriction weights and coarsening directions for each grid */
   for (lz = 0; lz <= (num_levels[2]-1); lz++)
   {
      for (ly = 0; ly <= (num_levels[1]-1); ly++)
      {
         for (lx = 0; lx <= (num_levels[0]-1); lx ++)
         {
            restrict_count = 0;
            hypre_SparseMSGComputeArrayIndex(
               lx,ly,lz,num_levels[0],num_levels[1],index);
            index *= 3;
            if (lx > 0) 
            {
               restrict_count++;
               restrict_weights[index] = 1;
            }
            else {restrict_weights[index] = 0;}
            if (ly > 0)
            {
               restrict_count++;
               restrict_weights[index + 1] = 1;
            }
            else {restrict_weights[index + 1] = 0;}
            if (lz > 0)
            {
               restrict_count++;
               restrict_weights[index + 2] = 1;
            }
            else {restrict_weights[index + 2] = 0;}
            
            if (restrict_count > 0)
            {
               for (l = index; l <= (index + 2); l++)
               {
                  restrict_weights[l] /= (double) restrict_count;
               }
            }
         }
      }
   }  
   return ierr;
}

int
hypre_SparseMSGComputeInterpWeights( int    *num_levels,
                                     double *interp_weights )
{
   int lx, ly, lz, l;
   int index;
   double interp_count;
   int ierr = 0;

   /* Compute interpolation weights and coarsening directions for each grid */
   for (lz = 0; lz <= (num_levels[2]-1); lz++)
   {
      for (ly = 0; ly <= (num_levels[1]-1); ly++)
      {
         for (lx = 0; lx <= (num_levels[0]-1); lx ++)
         {
            interp_count = 0.0;
            hypre_SparseMSGComputeArrayIndex(
               lx,ly,lz,num_levels[0],num_levels[1],index);
            index *= 3;
            if (lx < (num_levels[0]-1)) 
            {
               interp_weights[index] = 1/(pow(2,4*lx));
               interp_count += interp_weights[index];
            }
            else {interp_weights[index] = 0.0;}
            if (ly < (num_levels[1]-1))
            {
               interp_weights[index + 1] = 1/(pow(2,4*ly));
               interp_count += interp_weights[index + 1];
            }
            else {interp_weights[index + 1] = 0.0;}
            if (lz < (num_levels[2]-1))
            {
               interp_weights[index + 2] = 1/(pow(2,4*lz));
               interp_count += interp_weights[index + 2];
            }
            else {interp_weights[index + 2] = 0.0;}
            
            if (interp_count > 0.0)
            {
               for (l = index; l <= (index + 2); l++)
               {
                  interp_weights[l] /= interp_count;
               }
            }
         }
      }
   }  
   return ierr;
}
