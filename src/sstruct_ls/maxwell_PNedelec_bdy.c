/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.7 $
 ***********************************************************************EHEADER*/




#include "headers.h"

/*--------------------------------------------------------------------------
 * Finds the boundary boxes for all var_grids in pgrid. Use the cell grid
 * to determine the boundary.
 * bdry[n_cellboxes, nvars+1]= boxarrayarray ptr.: hypre_BoxArrayArray ***bdry.
 * bdry[n_cellboxes, 0] is the cell-centred box.
 * Each box_arrayarray: for each variable, there are a max of 2*(ndim-1)
 * box_arrays (e.g., in 3d, the x_edges on the boundary can be the two
 * z_faces & the two y_faces of the boundary). Each of these box_arrays
 * consists of boxes that can be on the boundary.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_Maxwell_PNedelec_Bdy( hypre_StructGrid       *cell_grid,
                            hypre_SStructPGrid     *pgrid,
                            hypre_BoxArrayArray ****bdry_ptr )
{

   HYPRE_Int ierr = 0;

   HYPRE_Int              nvars    = hypre_SStructPGridNVars(pgrid);

   hypre_BoxArrayArray   *cellgrid_bdry;
   hypre_BoxArrayArray ***bdry;
   hypre_BoxArray        *box_array, *box_array2;
   hypre_BoxArray        *cell_boxes;
   hypre_Box             *box, *bdy_box, *shifted_box;

   HYPRE_Int              ndim     = hypre_SStructPGridNDim(pgrid);

   HYPRE_SStructVariable *vartypes = hypre_SStructPGridVarTypes(pgrid);
   hypre_Index            varoffset, ishift, jshift, kshift;
   hypre_Index            lower, upper;

   HYPRE_Int             *flag;
   HYPRE_Int              i, j, k, t, nboxes, bdy;

   hypre_SetIndex(ishift, 1, 0, 0);
   hypre_SetIndex(jshift, 0, 1, 0);
   hypre_SetIndex(kshift, 0, 0, 1);

   cell_boxes= hypre_StructGridBoxes(cell_grid);
   nboxes    = hypre_BoxArraySize(cell_boxes);

   bdry= hypre_TAlloc(hypre_BoxArrayArray **, nboxes);
   shifted_box= hypre_BoxCreate();

   hypre_ForBoxI(j, cell_boxes)
   {
      box= hypre_BoxArrayBox(cell_boxes, j);

     /* find the cellgrid boundaries of box if there are any. */
      cellgrid_bdry= hypre_BoxArrayArrayCreate(2*ndim); 
      flag= hypre_CTAlloc(HYPRE_Int, 2*ndim);
      bdy = 0;

      for (i= 0; i< ndim; i++)
      {
         hypre_BoxBoundaryDG(box, cell_grid, 
                             hypre_BoxArrayArrayBoxArray(cellgrid_bdry, 2*i),
                             hypre_BoxArrayArrayBoxArray(cellgrid_bdry, 2*i+1),
                             i);
         if (hypre_BoxArraySize(hypre_BoxArrayArrayBoxArray(cellgrid_bdry, 2*i)))
         {
            flag[2*i]= 1;
            bdy++;
         }

         if (hypre_BoxArraySize(hypre_BoxArrayArrayBoxArray(cellgrid_bdry, 2*i+1)))
         {
            flag[2*i+1]= 1;
            bdy++;
         }
      }

     /* There are boundary boxes. Every variable of pgrid will have some */
      if (bdy)
      {
         bdry[j]= hypre_TAlloc(hypre_BoxArrayArray *, nvars+1); 
        
        /* keep the cell-centred boxarrayarray of boundaries */
         bdry[j][0]= hypre_BoxArrayArrayDuplicate(cellgrid_bdry); 

         k= 2*(ndim-1);  /* 3-d requires 4 boundary faces to be checked */
         for (i= 0; i< nvars; i++)
         {
            bdry[j][i+1]= hypre_BoxArrayArrayCreate(k); /* one for +/- directions */
         }
   
         for (i= 0; i< nvars; i++)
         {
            t= vartypes[i];
            hypre_SStructVariableGetOffset(vartypes[i], ndim, varoffset);

            switch(t)
            {
               case 2: /* xface, boundary i= lower, upper */
               {
                  if (flag[0]) /* boundary i= lower */
                  {
                     box_array = hypre_BoxArrayArrayBoxArray(cellgrid_bdry, 0);
                     box_array2= hypre_BoxArrayArrayBoxArray(bdry[j][i+1], 0);
                     hypre_ForBoxI(k, box_array)
                     {
                        bdy_box= hypre_BoxArrayBox(box_array, k);
                   
                       /* bdry boxes */
                        hypre_CopyIndex(hypre_BoxIMin(bdy_box), lower);
                        hypre_CopyIndex(hypre_BoxIMax(bdy_box), upper);
                        hypre_SubtractIndex(lower, varoffset, lower);
                        hypre_SubtractIndex(upper, varoffset, upper);
  
                        hypre_BoxSetExtents(shifted_box, lower, upper);
                        hypre_AppendBox(shifted_box, box_array2);
                     }
                  }

                  if (flag[1]) /* boundary i= upper */
                  {
                     box_array= hypre_BoxArrayArrayBoxArray(cellgrid_bdry, 1);
                     box_array2= hypre_BoxArrayArrayBoxArray(bdry[j][i+1], 1);
                     hypre_ForBoxI(k, box_array)
                     {
                        bdy_box= hypre_BoxArrayBox(box_array, k);

                       /* bdry boxes */
                        hypre_CopyIndex(hypre_BoxIMin(bdy_box), lower);
                        hypre_CopyIndex(hypre_BoxIMax(bdy_box), upper);
                                                                                                         
                        hypre_BoxSetExtents(shifted_box, lower, upper);
                        hypre_AppendBox(shifted_box, box_array2);
                     }
                  }

                  break;
               }

               case 3: /* yface, boundary j= lower, upper */
               {
                  if (flag[2]) /* boundary j= lower */
                  {
                     box_array = hypre_BoxArrayArrayBoxArray(cellgrid_bdry, 2);
                     box_array2= hypre_BoxArrayArrayBoxArray(bdry[j][i+1], 0);
                     hypre_ForBoxI(k, box_array)
                     {
                        bdy_box= hypre_BoxArrayBox(box_array, k);
                   
                       /* bdry boxes */
                        hypre_CopyIndex(hypre_BoxIMin(bdy_box), lower);
                        hypre_CopyIndex(hypre_BoxIMax(bdy_box), upper);
                        hypre_SubtractIndex(lower, varoffset, lower);
                        hypre_SubtractIndex(upper, varoffset, upper);
  
                        hypre_BoxSetExtents(shifted_box, lower, upper);
                        hypre_AppendBox(shifted_box, box_array2);
                     }
                  }

                  if (flag[3]) /* boundary j= upper */
                  {
                     box_array= hypre_BoxArrayArrayBoxArray(cellgrid_bdry, 3);
                     box_array2= hypre_BoxArrayArrayBoxArray(bdry[j][i+1], 1);
                     hypre_ForBoxI(k, box_array)
                     {
                        bdy_box= hypre_BoxArrayBox(box_array, k);
                                                                                                         
                       /* bdry boxes */
                        hypre_CopyIndex(hypre_BoxIMin(bdy_box), lower);
                        hypre_CopyIndex(hypre_BoxIMax(bdy_box), upper);

                        hypre_BoxSetExtents(shifted_box, lower, upper);
                        hypre_AppendBox(shifted_box, box_array2);
                     }
                  }

                  break;
               }

               case 5: /* xedge, boundary z_faces & y_faces */
               {
                  if (flag[4]) /* boundary k= lower zface*/
                  {
                     box_array = hypre_BoxArrayArrayBoxArray(cellgrid_bdry, 4);
                     box_array2= hypre_BoxArrayArrayBoxArray(bdry[j][i+1], 0);
                     hypre_ForBoxI(k, box_array)
                     {
                        bdy_box= hypre_BoxArrayBox(box_array, k);
                   
                       /* bdry boxes */
                        hypre_CopyIndex(hypre_BoxIMin(bdy_box), lower);
                        hypre_CopyIndex(hypre_BoxIMax(bdy_box), upper);
                        hypre_SubtractIndex(lower, varoffset, lower);
                        hypre_SubtractIndex(upper, kshift, upper);
  
                        hypre_BoxSetExtents(shifted_box, lower, upper);
                        hypre_AppendBox(shifted_box, box_array2);
                     }
                  }

                  if (flag[5]) /* boundary k= upper zface*/
                  {
                     box_array = hypre_BoxArrayArrayBoxArray(cellgrid_bdry, 5);
                     box_array2= hypre_BoxArrayArrayBoxArray(bdry[j][i+1], 1);
                     hypre_ForBoxI(k, box_array)
                     {
                        bdy_box= hypre_BoxArrayBox(box_array, k);
                   
                       /* bdry boxes */
                        hypre_CopyIndex(hypre_BoxIMin(bdy_box), lower);
                        hypre_CopyIndex(hypre_BoxIMax(bdy_box), upper);
                        hypre_SubtractIndex(lower, jshift, lower);
  
                        hypre_BoxSetExtents(shifted_box, lower, upper);
                        hypre_AppendBox(shifted_box, box_array2);
                     }
                  }

                  if (flag[2]) /* boundary j= lower yface*/
                  {
                     box_array = hypre_BoxArrayArrayBoxArray(cellgrid_bdry, 2);
                     box_array2= hypre_BoxArrayArrayBoxArray(bdry[j][i+1], 2);
                     hypre_ForBoxI(k, box_array)
                     {
                        bdy_box= hypre_BoxArrayBox(box_array, k);
                                                                                                                        
                       /* bdry boxes */
                        hypre_CopyIndex(hypre_BoxIMin(bdy_box), lower);
                        hypre_CopyIndex(hypre_BoxIMax(bdy_box), upper);
                        hypre_SubtractIndex(lower, varoffset, lower);
                        hypre_SubtractIndex(upper, jshift, upper);
                                                                                                                        
                        hypre_BoxSetExtents(shifted_box, lower, upper);
                        hypre_AppendBox(shifted_box, box_array2);
                     }
                  }
                                                                                                                        
                  if (flag[3]) /* boundary j= upper yface*/
                  {
                     box_array = hypre_BoxArrayArrayBoxArray(cellgrid_bdry, 3);
                     box_array2= hypre_BoxArrayArrayBoxArray(bdry[j][i+1], 3);
                     hypre_ForBoxI(k, box_array)
                     {
                        bdy_box= hypre_BoxArrayBox(box_array, k);

                       /* bdry boxes */
                        hypre_CopyIndex(hypre_BoxIMin(bdy_box), lower);
                        hypre_CopyIndex(hypre_BoxIMax(bdy_box), upper);
                        hypre_SubtractIndex(lower, kshift, lower);
                                                                                                                        
                        hypre_BoxSetExtents(shifted_box, lower, upper);
                        hypre_AppendBox(shifted_box, box_array2);
                     }
                  }
                  break;
               }

               case 6: /* yedge, boundary z_faces & x_faces */
               {
                  if (flag[4]) /* boundary k= lower zface*/
                  {
                     box_array = hypre_BoxArrayArrayBoxArray(cellgrid_bdry, 4);
                     box_array2= hypre_BoxArrayArrayBoxArray(bdry[j][i+1], 0);
                     hypre_ForBoxI(k, box_array)
                     {
                        bdy_box= hypre_BoxArrayBox(box_array, k);

                       /* bdry boxes */
                        hypre_CopyIndex(hypre_BoxIMin(bdy_box), lower);
                        hypre_CopyIndex(hypre_BoxIMax(bdy_box), upper);
                        hypre_SubtractIndex(lower, varoffset, lower);
                        hypre_SubtractIndex(upper, kshift, upper);
                                                                                                                        
                        hypre_BoxSetExtents(shifted_box, lower, upper);
                        hypre_AppendBox(shifted_box, box_array2);
                     }
                  }
                                                                                                                        
                  if (flag[5]) /* boundary k= upper zface*/
                  {
                     box_array = hypre_BoxArrayArrayBoxArray(cellgrid_bdry, 5);
                     box_array2= hypre_BoxArrayArrayBoxArray(bdry[j][i+1], 1);
                     hypre_ForBoxI(k, box_array)
                     {
                        bdy_box= hypre_BoxArrayBox(box_array, k);

                       /* bdry boxes */
                        hypre_CopyIndex(hypre_BoxIMin(bdy_box), lower);
                        hypre_CopyIndex(hypre_BoxIMax(bdy_box), upper);
                        hypre_SubtractIndex(lower, ishift, lower);
                                                                                                                        
                        hypre_BoxSetExtents(shifted_box, lower, upper);
                        hypre_AppendBox(shifted_box, box_array2);
                     }
                  }
                                                                                                                        
                  if (flag[0]) /* boundary i= lower xface*/
                  {
                     box_array = hypre_BoxArrayArrayBoxArray(cellgrid_bdry, 0);
                     box_array2= hypre_BoxArrayArrayBoxArray(bdry[j][i+1], 2);
                     hypre_ForBoxI(k, box_array)
                     {
                        bdy_box= hypre_BoxArrayBox(box_array, k);

                       /* bdry boxes */
                        hypre_CopyIndex(hypre_BoxIMin(bdy_box), lower);
                        hypre_CopyIndex(hypre_BoxIMax(bdy_box), upper);
                        hypre_SubtractIndex(lower, varoffset, lower);
                        hypre_SubtractIndex(upper, ishift, upper);
                                                                                                                        
                        hypre_BoxSetExtents(shifted_box, lower, upper);
                        hypre_AppendBox(shifted_box, box_array2);
                     }
                  }
                                                                                                                        
                  if (flag[1]) /* boundary i= upper xface*/
                  {
                     box_array = hypre_BoxArrayArrayBoxArray(cellgrid_bdry, 1);
                     box_array2= hypre_BoxArrayArrayBoxArray(bdry[j][i+1], 3);
                     hypre_ForBoxI(k, box_array)
                     {
                        bdy_box= hypre_BoxArrayBox(box_array, k);
                                                                                                                        
                       /* bdry boxes */
                        hypre_CopyIndex(hypre_BoxIMin(bdy_box), lower);
                        hypre_CopyIndex(hypre_BoxIMax(bdy_box), upper);
                        hypre_SubtractIndex(lower, kshift, lower);

                        hypre_BoxSetExtents(shifted_box, lower, upper);
                        hypre_AppendBox(shifted_box, box_array2);
                     }
                  }

                  break;
               }

               case 7: /* zedge, boundary y_faces & x_faces */
               {
                  if (flag[2]) /* boundary j= lower yface*/
                  {
                      box_array = hypre_BoxArrayArrayBoxArray(cellgrid_bdry, 2);
                      box_array2= hypre_BoxArrayArrayBoxArray(bdry[j][i+1], 0);
                      hypre_ForBoxI(k, box_array)
                      {
                         bdy_box= hypre_BoxArrayBox(box_array, k);
                                                                                                                        
                        /* bdry boxes */
                         hypre_CopyIndex(hypre_BoxIMin(bdy_box), lower);
                         hypre_CopyIndex(hypre_BoxIMax(bdy_box), upper);
                         hypre_SubtractIndex(lower, varoffset, lower);
                         hypre_SubtractIndex(upper, jshift, upper);

                         hypre_BoxSetExtents(shifted_box, lower, upper);
                         hypre_AppendBox(shifted_box, box_array2);
                      }
                   }

                   if (flag[3]) /* boundary j= upper yface*/
                   {
                      box_array = hypre_BoxArrayArrayBoxArray(cellgrid_bdry, 3);
                      box_array2= hypre_BoxArrayArrayBoxArray(bdry[j][i+1], 1);
                      hypre_ForBoxI(k, box_array)
                      {
                         bdy_box= hypre_BoxArrayBox(box_array, k);

                        /* bdry boxes */
                         hypre_CopyIndex(hypre_BoxIMin(bdy_box), lower);
                         hypre_CopyIndex(hypre_BoxIMax(bdy_box), upper);
                         hypre_SubtractIndex(lower, ishift, lower);

                         hypre_BoxSetExtents(shifted_box, lower, upper);
                         hypre_AppendBox(shifted_box, box_array2);
                      }
                   }

                   if (flag[0]) /* boundary i= lower xface*/
                   {
                      box_array = hypre_BoxArrayArrayBoxArray(cellgrid_bdry, 0);
                      box_array2= hypre_BoxArrayArrayBoxArray(bdry[j][i+1], 2);
                      hypre_ForBoxI(k, box_array)
                      {
                         bdy_box= hypre_BoxArrayBox(box_array, k);

                        /* bdry boxes */
                         hypre_CopyIndex(hypre_BoxIMin(bdy_box), lower);
                         hypre_CopyIndex(hypre_BoxIMax(bdy_box), upper);
                         hypre_SubtractIndex(lower, varoffset, lower);
                         hypre_SubtractIndex(upper, ishift, upper);

                         hypre_BoxSetExtents(shifted_box, lower, upper);
                         hypre_AppendBox(shifted_box, box_array2);
                      }
                   }

                   if (flag[1]) /* boundary i= upper xface*/
                   {
                      box_array = hypre_BoxArrayArrayBoxArray(cellgrid_bdry, 1);
                      box_array2= hypre_BoxArrayArrayBoxArray(bdry[j][i+1], 3);
                      hypre_ForBoxI(k, box_array)
                      {
                         bdy_box= hypre_BoxArrayBox(box_array, k);

                        /* bdry boxes */
                         hypre_CopyIndex(hypre_BoxIMin(bdy_box), lower);
                         hypre_CopyIndex(hypre_BoxIMax(bdy_box), upper);
                         hypre_SubtractIndex(lower, jshift, lower);

                         hypre_BoxSetExtents(shifted_box, lower, upper);
                         hypre_AppendBox(shifted_box, box_array2);
                      }
                   }
                   break;
               }

            }  /* switch(t) */
         }     /* for (i= 0; i< nvars; i++) */
      }        /* if (bdy) */

      else
      {
        /* make an empty ptr of boxarrayarrays to avoid memory leaks when
           destroying bdry later. */
         bdry[j]= hypre_TAlloc(hypre_BoxArrayArray *, nvars+1); 
         for (i= 0; i< nvars+1; i++)
         {
            bdry[j][i]= hypre_BoxArrayArrayCreate(0);
         }
      }

      hypre_BoxArrayArrayDestroy(cellgrid_bdry); 
      hypre_TFree(flag);
   }  /* hypre_ForBoxI(j, cell_boxes) */

   hypre_BoxDestroy(shifted_box);

  *bdry_ptr     = bdry;

   return ierr;
}

