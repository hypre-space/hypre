/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Member functions for hypre_StructMatrix class.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_StructMatrixExtractPointerByIndex
 *    Returns pointer to data for stencil entry coresponding to
 *    `index' in `matrix'. If the index does not exist in the matrix's
 *    stencil, the NULL pointer is returned. 
 *--------------------------------------------------------------------------*/
 
double *
hypre_StructMatrixExtractPointerByIndex( hypre_StructMatrix *matrix,
                                         int                 b,
                                         hypre_Index         index  )
{
   hypre_StructStencil   *stencil;
   int                    rank;

   stencil = hypre_StructMatrixStencil(matrix);
   rank = hypre_StructStencilElementRank( stencil, index );

   if ( rank >= 0 )
      return hypre_StructMatrixBoxData(matrix, b, rank);
   else
      return NULL;  /* error - invalid index */
}

/*--------------------------------------------------------------------------
 * hypre_StructMatrixCreate
 *--------------------------------------------------------------------------*/

hypre_StructMatrix *
hypre_StructMatrixCreate( MPI_Comm             comm,
                          hypre_StructGrid    *grid,
                          hypre_StructStencil *user_stencil )
{
   hypre_StructMatrix  *matrix;

   int                  ndim             = hypre_StructGridDim(grid);
   int                  i;

   matrix = hypre_CTAlloc(hypre_StructMatrix, 1);

   hypre_StructMatrixComm(matrix)        = comm;
   hypre_StructGridRef(grid, &hypre_StructMatrixGrid(matrix));
   hypre_StructMatrixUserStencil(matrix) =
      hypre_StructStencilRef(user_stencil);
   hypre_StructMatrixDataAlloced(matrix) = 1;
   hypre_StructMatrixRefCount(matrix)    = 1;

   /* set defaults */
   hypre_StructMatrixSymmetric(matrix) = 0;
   hypre_StructMatrixConstantCoefficient(matrix) = 0;
   for (i = 0; i < 6; i++)
   {
      hypre_StructMatrixNumGhost(matrix)[i] = 0;
      hypre_StructMatrixAddNumGhost(matrix)[i]= 0;
   }

   for (i = 0; i < ndim; i++)
   {
      hypre_StructMatrixNumGhost(matrix)[2*i] = 1;
      hypre_StructMatrixNumGhost(matrix)[2*i+1] = 1;

      hypre_StructMatrixAddNumGhost(matrix)[2*i]= 1;
      hypre_StructMatrixAddNumGhost(matrix)[2*i+1]= 1;
   }

   hypre_StructMatrixOffProcAdd(matrix) = 0;

   return matrix;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatrixRef
 *--------------------------------------------------------------------------*/

hypre_StructMatrix *
hypre_StructMatrixRef( hypre_StructMatrix *matrix )
{
   hypre_StructMatrixRefCount(matrix) ++;

   return matrix;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatrixDestroy
 *--------------------------------------------------------------------------*/

int 
hypre_StructMatrixDestroy( hypre_StructMatrix *matrix )
{
   int  i;
   int  ierr = 0;

   if (matrix)
   {
      hypre_StructMatrixRefCount(matrix) --;
      if (hypre_StructMatrixRefCount(matrix) == 0)
      {
         if (hypre_StructMatrixDataAlloced(matrix))
         {
            hypre_SharedTFree(hypre_StructMatrixData(matrix));
         }
         hypre_CommPkgDestroy(hypre_StructMatrixCommPkg(matrix));
         
         hypre_ForBoxI(i, hypre_StructMatrixDataSpace(matrix))
            hypre_TFree(hypre_StructMatrixDataIndices(matrix)[i]);
         hypre_TFree(hypre_StructMatrixDataIndices(matrix));
         
         hypre_BoxArrayDestroy(hypre_StructMatrixDataSpace(matrix));
         
         hypre_TFree(hypre_StructMatrixSymmElements(matrix));
         hypre_StructStencilDestroy(hypre_StructMatrixUserStencil(matrix));
         hypre_StructStencilDestroy(hypre_StructMatrixStencil(matrix));
         hypre_StructGridDestroy(hypre_StructMatrixGrid(matrix));
         
         hypre_TFree(matrix);
      }
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatrixInitializeShell
 *--------------------------------------------------------------------------*/

int 
hypre_StructMatrixInitializeShell( hypre_StructMatrix *matrix )
{
   int    ierr = 0;

   hypre_StructGrid     *grid;

   hypre_StructStencil  *user_stencil;
   hypre_StructStencil  *stencil;
   hypre_Index          *stencil_shape;
   int                   stencil_size;
   int                   num_values;
   int                  *symm_elements;
   int                  constant_coefficient;
                    
   int                  *num_ghost;
   int                   extra_ghost[] = {0, 0, 0, 0, 0, 0};
 
   hypre_BoxArray       *data_space;
   hypre_BoxArray       *boxes;
   hypre_Box            *box;
   hypre_Box            *data_box;

   int                 **data_indices;
   int                   data_size;
   int                   data_box_volume;
                    
   int                   i, j, d;

   grid = hypre_StructMatrixGrid(matrix);

   /*-----------------------------------------------------------------------
    * Set up stencil and num_values:
    *
    * If the matrix is symmetric, then the stencil is a "symmetrized"
    * version of the user's stencil.  If the matrix is not symmetric,
    * then the stencil is the same as the user's stencil.
    * 
    * The `symm_elements' array is used to determine what data is
    * explicitely stored (symm_elements[i] < 0) and what data does is
    * not explicitely stored (symm_elements[i] >= 0), but is instead
    * stored as the transpose coefficient at a neighboring grid point.
    *-----------------------------------------------------------------------*/

   if (hypre_StructMatrixStencil(matrix) == NULL)
   {
      user_stencil = hypre_StructMatrixUserStencil(matrix);

      if (hypre_StructMatrixSymmetric(matrix))
      {
         /* store only symmetric stencil entry data */
         hypre_StructStencilSymmetrize(user_stencil, &stencil, &symm_elements);
         num_values = ( hypre_StructStencilSize(stencil) + 1 ) / 2;
      }
      else
      {
         /* store all stencil entry data */
         stencil = hypre_StructStencilRef(user_stencil);
         num_values = hypre_StructStencilSize(stencil);
         symm_elements = hypre_TAlloc(int, num_values);
         for (i = 0; i < num_values; i++)
         {
            symm_elements[i] = -1;
         }
      }

      hypre_StructMatrixStencil(matrix)      = stencil;
      hypre_StructMatrixSymmElements(matrix) = symm_elements;
      hypre_StructMatrixNumValues(matrix)    = num_values;
   }

   /*-----------------------------------------------------------------------
    * Set ghost-layer size for symmetric storage
    *   - All stencil coeffs are to be available at each point in the
    *     grid, as well as in the user-specified ghost layer.
    *-----------------------------------------------------------------------*/

   num_ghost     = hypre_StructMatrixNumGhost(matrix);
   stencil       = hypre_StructMatrixStencil(matrix);
   stencil_shape = hypre_StructStencilShape(stencil);
   stencil_size  = hypre_StructStencilSize(stencil);
   symm_elements = hypre_StructMatrixSymmElements(matrix);

   for (i = 0; i < stencil_size; i++)
   {
      if (symm_elements[i] >= 0)
      {
         for (d = 0; d < 3; d++)
         {
            extra_ghost[2*d] =
               hypre_max(extra_ghost[2*d], -hypre_IndexD(stencil_shape[i], d));
            extra_ghost[2*d + 1] =
               hypre_max(extra_ghost[2*d + 1],  hypre_IndexD(stencil_shape[i], d));
         }
      }
   }

   for (d = 0; d < 3; d++)
   {
      num_ghost[2*d]     += extra_ghost[2*d];
      num_ghost[2*d + 1] += extra_ghost[2*d + 1];
   }

   /*-----------------------------------------------------------------------
    * Set up data_space
    *-----------------------------------------------------------------------*/

   if (hypre_StructMatrixDataSpace(matrix) == NULL)
   {
      boxes = hypre_StructGridBoxes(grid);
      data_space = hypre_BoxArrayCreate(hypre_BoxArraySize(boxes));

      hypre_ForBoxI(i, boxes)
         {
            box = hypre_BoxArrayBox(boxes, i);
            data_box = hypre_BoxArrayBox(data_space, i);

            hypre_CopyBox(box, data_box);
            for (d = 0; d < 3; d++)
            {
               hypre_BoxIMinD(data_box, d) -= num_ghost[2*d];
               hypre_BoxIMaxD(data_box, d) += num_ghost[2*d + 1];
            }
         }

      hypre_StructMatrixDataSpace(matrix) = data_space;
   }

   /*-----------------------------------------------------------------------
    * Set up data_indices array and data-size
    *-----------------------------------------------------------------------*/

   if (hypre_StructMatrixDataIndices(matrix) == NULL)
   {
      data_space = hypre_StructMatrixDataSpace(matrix);
      data_indices = hypre_CTAlloc(int *, hypre_BoxArraySize(data_space));
      constant_coefficient = hypre_StructMatrixConstantCoefficient(matrix);

      data_size = 0;
      if ( constant_coefficient==0 )
      {
         hypre_ForBoxI(i, data_space)
            {
               data_box = hypre_BoxArrayBox(data_space, i);
               data_box_volume  = hypre_BoxVolume(data_box);

               data_indices[i] = hypre_CTAlloc(int, stencil_size);

               /* set pointers for "stored" coefficients */
               for (j = 0; j < stencil_size; j++)
               {
                  if (symm_elements[j] < 0)
                  {
                     data_indices[i][j] = data_size;
                     data_size += data_box_volume;
                  }
               }

               /* set pointers for "symmetric" coefficients */
               for (j = 0; j < stencil_size; j++)
               {
                  if (symm_elements[j] >= 0)
                  {
                     data_indices[i][j] = data_indices[i][symm_elements[j]] +
                        hypre_BoxOffsetDistance(data_box, stencil_shape[j]);
                  }
               }
            }
      }
      else if ( constant_coefficient==1 )
      {
      
         hypre_ForBoxI(i, data_space)
            {
               data_box = hypre_BoxArrayBox(data_space, i);
               data_box_volume  = hypre_BoxVolume(data_box);

               data_indices[i] = hypre_CTAlloc(int, stencil_size);

               /* set pointers for "stored" coefficients */
               for (j = 0; j < stencil_size; j++)
               {
                  if (symm_elements[j] < 0)
                  {
                     data_indices[i][j] = data_size;
                     ++data_size;
                  }
               }

               /* set pointers for "symmetric" coefficients */
               for (j = 0; j < stencil_size; j++)
               {
                  if (symm_elements[j] >= 0)
                  {
                     data_indices[i][j] = data_indices[i][symm_elements[j]];
                  }
               }
            }
      }
      else
      {
         hypre_assert( constant_coefficient == 2 );
         data_size += stencil_size;  /* all constant coefficients at the beginning */
         /* ... this allocates a little more space than is absolutely necessary */
         hypre_ForBoxI(i, data_space)
            {
               data_box = hypre_BoxArrayBox(data_space, i);
               data_box_volume  = hypre_BoxVolume(data_box);

               data_indices[i] = hypre_CTAlloc(int, stencil_size);

               /* set pointers for "stored" coefficients */
               for (j = 0; j < stencil_size; j++)
               {
                  if (symm_elements[j] < 0)
                  {
                     if (
                        hypre_IndexX(stencil_shape[j])==0 &&
                        hypre_IndexY(stencil_shape[j])==0 &&
                        hypre_IndexZ(stencil_shape[j])==0 )  /* diagonal, variable coefficient */
                     {
                        data_indices[i][j] = data_size;
                        data_size += data_box_volume;
                     }
                     else /* off-diagonal, constant coefficient */
                     {
                        data_indices[i][j] = j;
                     }
                  }
               }

               /* set pointers for "symmetric" coefficients */
               for (j = 0; j < stencil_size; j++)
               {
                  if (symm_elements[j] >= 0)
                  {
                     if (
                        hypre_IndexX(stencil_shape[j])==0 &&
                        hypre_IndexY(stencil_shape[j])==0 &&
                        hypre_IndexZ(stencil_shape[j])==0 )  /* diagonal, variable coefficient */
                     {
                        data_indices[i][j] = data_indices[i][symm_elements[j]] +
                           hypre_BoxOffsetDistance(data_box, stencil_shape[j]);
                     }
                     else /* off-diagonal, constant coefficient */
                     {
                        data_indices[i][j] = data_indices[i][symm_elements[j]];
                     }
                  }
               }
            }
      }

      hypre_StructMatrixDataIndices(matrix) = data_indices;
      hypre_StructMatrixDataSize(matrix)    = data_size;
   }

   /*-----------------------------------------------------------------------
    * Set total number of nonzero coefficients
    * For constant coefficients, this is unrelated to the amount of data
    * actually stored.
    *-----------------------------------------------------------------------*/

   hypre_StructMatrixGlobalSize(matrix) =
      hypre_StructGridGlobalSize(grid) * stencil_size;

   /*-----------------------------------------------------------------------
    * Return
    *-----------------------------------------------------------------------*/

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatrixInitializeData
 *--------------------------------------------------------------------------*/

int
hypre_StructMatrixInitializeData( hypre_StructMatrix *matrix,
                                  double             *data   )
{
   int             ierr = 0;

   hypre_BoxArray      *data_boxes;
   hypre_Box           *data_box;
   hypre_Index          loop_size;
   hypre_Index          index;
   hypre_IndexRef       start;
   hypre_Index          stride;
   double              *datap;
   int                  datai;
   int                  constant_coefficient;
   int                  i;
   int                  loopi, loopj, loopk;

   hypre_StructMatrixData(matrix) = data;
   hypre_StructMatrixDataAlloced(matrix) = 0;
   constant_coefficient = hypre_StructMatrixConstantCoefficient(matrix);

   /*-------------------------------------------------
    * If the matrix has a diagonal, set these entries
    * to 1 inside the grid_boxes.  Ghostvalues will 
    * be set to 1 in the assembly. This reduces the 
    * complexity of many computations by eliminating 
    * divide-by-zero in the ghost region.
    *-------------------------------------------------*/

   hypre_SetIndex(index, 0, 0, 0);
   hypre_SetIndex(stride, 1, 1, 1);

   data_boxes = hypre_StructMatrixDataSpace(matrix);
   hypre_ForBoxI(i, data_boxes)
      {
         datap = hypre_StructMatrixExtractPointerByIndex(matrix, i, index);

         if (datap)
         {
            data_box = hypre_BoxArrayBox(data_boxes, i);
            start = hypre_BoxIMin(data_box);

            if ( constant_coefficient==1 )
            {
               datai = hypre_CCBoxIndexRank(data_box,start);
               datap[datai] = 1.0;
            }
            else
               /* either fully variable coefficient matrix, constant_coefficient=0,
                  or variable diagonal (otherwise constant) coefficient matrix,
                  constant_coefficient=2 */
            {
               hypre_BoxGetSize(data_box, loop_size);

               hypre_BoxLoop1Begin(loop_size,
                                   data_box, start, stride, datai);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,datai
#include "hypre_box_smp_forloop.h"
               hypre_BoxLoop1For(loopi, loopj, loopk, datai)
                  {
                     datap[datai] = 1.0;        
                  }
               hypre_BoxLoop1End(datai);
            }
         }
      }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatrixInitialize
 *--------------------------------------------------------------------------*/
int 
hypre_StructMatrixInitialize( hypre_StructMatrix *matrix )
{
   int    ierr = 0;

   double *data;

   ierr = hypre_StructMatrixInitializeShell(matrix);

   data = hypre_StructMatrixData(matrix);
   data = hypre_SharedCTAlloc(double, hypre_StructMatrixDataSize(matrix));
   hypre_StructMatrixInitializeData(matrix, data);
   hypre_StructMatrixDataAlloced(matrix) = 1;

   return ierr;
}

/*--------------------------------------------------------------------------
 * (action > 0): add-to values
 * (action = 0): set values
 * (action < 0): get values
 * should not be called to set a constant-coefficient part of the matrix,
 *   call hypre_StructMatrixSetConstantValues instead
 *--------------------------------------------------------------------------*/

int 
hypre_StructMatrixSetValues( hypre_StructMatrix *matrix,
                             hypre_Index         grid_index,
                             int                 num_stencil_indices,
                             int                *stencil_indices,
                             double             *values,
                             int                 action )
{
   int    ierr = 0;
   MPI_Comm            comm= hypre_StructMatrixComm(matrix);

   hypre_BoxArray     *boxes;
   hypre_Box          *box;
   hypre_Index         center_index;
   hypre_StructStencil *stencil;
   int                 center_rank;
   int                 constant_coefficient;

   double             *matp;

   int                 i, s, found;
   int                 true = 1;
   int                 false= 0;
 
   int                 nprocs;

   MPI_Comm_size(comm, &nprocs );

   boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(matrix));
   constant_coefficient = hypre_StructMatrixConstantCoefficient(matrix);

   found= true; /* index found will be set to false later on. This 
                   eliminates the constant_coefficient= 1 case correctly. */
   hypre_ForBoxI(i, boxes)
      {
         box = hypre_BoxArrayBox(boxes, i);

         if ( constant_coefficient==1 )
         {
            ++ierr;  /* call SetConstantValues instead */
            if (action > 0)
            {
               for (s = 0; s < num_stencil_indices; s++)
               {
                  matp = hypre_StructMatrixBoxData(matrix, i,
                                                        stencil_indices[s]);
                  *matp += values[s];
               }
            }
            else if (action > -1)
            {
               for (s = 0; s < num_stencil_indices; s++)
               {
                  matp = hypre_StructMatrixBoxData(matrix, i,
                                                        stencil_indices[s]);
                  *matp = values[s];
               }
            }
            else  /* action < 0 */
            {
               for (s = 0; s < num_stencil_indices; s++)
               {
                  matp = hypre_StructMatrixBoxData(matrix, i,
                                                        stencil_indices[s]);
                  values[s] = *matp;
               }
            }
         }
         else if ( constant_coefficient==2 )
         {
            hypre_SetIndex(center_index, 0, 0, 0);
            stencil = hypre_StructMatrixStencil(matrix);
            center_rank = hypre_StructStencilElementRank( stencil, center_index );

            found= false;
            if ( action > 0 )
            {
               for (s = 0; s < num_stencil_indices; s++)
               {
                  if ( stencil_indices[s] == center_rank )
                  {  /* center (diagonal), like constant_coefficient==0 */
                     if ((hypre_IndexX(grid_index) >= hypre_BoxIMinX(box)) &&
                         (hypre_IndexX(grid_index) <= hypre_BoxIMaxX(box)) &&
                         (hypre_IndexY(grid_index) >= hypre_BoxIMinY(box)) &&
                         (hypre_IndexY(grid_index) <= hypre_BoxIMaxY(box)) &&
                         (hypre_IndexZ(grid_index) >= hypre_BoxIMinZ(box)) &&
                         (hypre_IndexZ(grid_index) <= hypre_BoxIMaxZ(box))   )
                     {
                        matp = hypre_StructMatrixBoxDataValue(matrix, i,
                                                              stencil_indices[s],
                                                              grid_index);
                        *matp += values[s];
                        found= true;
                     }
                  }
                  else
                  {  /* non-center, like constant_coefficient==1 */
                     ++ierr;  /* should have called SetConstantValues */
                     matp = hypre_StructMatrixBoxData(matrix, i,
                                                      stencil_indices[s]);
                     *matp += values[s];
                     found= true;
                  }
               }
            }
            else if ( action > -1 )
            {
               for (s = 0; s < num_stencil_indices; s++)
               {
                  if ( stencil_indices[s] == center_rank )
                  {  /* center (diagonal), like constant_coefficient==0 */
                     if ((hypre_IndexX(grid_index) >= hypre_BoxIMinX(box)) &&
                         (hypre_IndexX(grid_index) <= hypre_BoxIMaxX(box)) &&
                         (hypre_IndexY(grid_index) >= hypre_BoxIMinY(box)) &&
                         (hypre_IndexY(grid_index) <= hypre_BoxIMaxY(box)) &&
                         (hypre_IndexZ(grid_index) >= hypre_BoxIMinZ(box)) &&
                         (hypre_IndexZ(grid_index) <= hypre_BoxIMaxZ(box))   )
                     {
                        matp = hypre_StructMatrixBoxDataValue(matrix, i,
                                                              stencil_indices[s],
                                                              grid_index);
                        *matp = values[s];
                        found= true;
                     }
                  }
                  else
                  {  /* non-center, like constant_coefficient==1 */
                     ++ierr;  /* should have called SetConstantValues */
                     matp = hypre_StructMatrixBoxData(matrix, i,
                                                      stencil_indices[s]);
                     *matp += values[s];
                     found= true;
                  }
               }
            }
            else  /* action<0 */
            {
               found= true; /* no need to set-off proc for get values */
               for (s = 0; s < num_stencil_indices; s++)
               {
                  if ( stencil_indices[s] == center_rank )
                  {  /* center (diagonal), like constant_coefficient==0 */
                     if ((hypre_IndexX(grid_index) >= hypre_BoxIMinX(box)) &&
                         (hypre_IndexX(grid_index) <= hypre_BoxIMaxX(box)) &&
                         (hypre_IndexY(grid_index) >= hypre_BoxIMinY(box)) &&
                         (hypre_IndexY(grid_index) <= hypre_BoxIMaxY(box)) &&
                         (hypre_IndexZ(grid_index) >= hypre_BoxIMinZ(box)) &&
                         (hypre_IndexZ(grid_index) <= hypre_BoxIMaxZ(box))   )
                     {
                        matp = hypre_StructMatrixBoxDataValue(matrix, i,
                                                              stencil_indices[s],
                                                              grid_index);
                        *matp += values[s];
                     }
                  }
                  else
                  {  /* non-center, like constant_coefficient==1 */
                     ++ierr;  /* should have called SetConstantValues */
                     matp = hypre_StructMatrixBoxData(matrix, i,
                                                      stencil_indices[s]);
                     values[s] = *matp;
                  }
               }
            }
         }
         else
            /* variable coefficient, constant_coefficient=0 */
         {
            found= false;
            if ((hypre_IndexX(grid_index) >= hypre_BoxIMinX(box)) &&
                (hypre_IndexX(grid_index) <= hypre_BoxIMaxX(box)) &&
                (hypre_IndexY(grid_index) >= hypre_BoxIMinY(box)) &&
                (hypre_IndexY(grid_index) <= hypre_BoxIMaxY(box)) &&
                (hypre_IndexZ(grid_index) >= hypre_BoxIMinZ(box)) &&
                (hypre_IndexZ(grid_index) <= hypre_BoxIMaxZ(box))   )
            {
               if (action > 0)
               {
                  for (s = 0; s < num_stencil_indices; s++)
                  {
                     matp = hypre_StructMatrixBoxDataValue(matrix, i,
                                                           stencil_indices[s],
                                                           grid_index);
                     *matp += values[s];
                  }
               }
               else if (action > -1)
               {
                  for (s = 0; s < num_stencil_indices; s++)
                  {
                     matp = hypre_StructMatrixBoxDataValue(matrix, i,
                                                           stencil_indices[s],
                                                           grid_index);
                     *matp = values[s];
                  }
               }
               else
               {
                  for (s = 0; s < num_stencil_indices; s++)
                  {
                     matp = hypre_StructMatrixBoxDataValue(matrix, i,
                                                           stencil_indices[s],
                                                           grid_index);
                     values[s] = *matp;
                  }
               }

               found= true;
            }
         }
      }

      /* if the index was not found on this proc and the user wants to ADD
         an off_proc value, see if it is along the ghostlayer. */
      if ((!found) && (action > 0) && (nprocs > 1))
      {
         hypre_Box  *orig_box;

         int        *add_num_ghost= hypre_StructMatrixAddNumGhost(matrix);
         int         j;

         hypre_ForBoxI(i, boxes)
         {
            orig_box = hypre_BoxArrayBox(boxes, i);
            box      = hypre_BoxDuplicate(orig_box);
            for (j= 0; j< 3; j++)
            {
               hypre_BoxIMin(box)[j]-= add_num_ghost[2*j];
               hypre_BoxIMax(box)[j]+= add_num_ghost[2*j+1];
            }

            if ( constant_coefficient==2 ) /* only center and action > 0 */
            {
               hypre_SetIndex(center_index, 0, 0, 0);
               stencil = hypre_StructMatrixStencil(matrix);
               center_rank = hypre_StructStencilElementRank( stencil, center_index );

               /* center (diagonal), like constant_coefficient==0 */
               if ((hypre_IndexX(grid_index) >= hypre_BoxIMinX(box)) &&
                   (hypre_IndexX(grid_index) <= hypre_BoxIMaxX(box)) &&
                   (hypre_IndexY(grid_index) >= hypre_BoxIMinY(box)) &&
                   (hypre_IndexY(grid_index) <= hypre_BoxIMaxY(box)) &&
                   (hypre_IndexZ(grid_index) >= hypre_BoxIMinZ(box)) &&
                   (hypre_IndexZ(grid_index) <= hypre_BoxIMaxZ(box))   )
               {
                  matp = hypre_StructMatrixBoxDataValue(matrix, i,
                                                        stencil_indices[center_rank],
                                                        grid_index);
                 *matp += values[s];
                  found= true;
               }
            }

            else
            /* variable coefficient, constant_coefficient=0 */
            {
               if ((hypre_IndexX(grid_index) >= hypre_BoxIMinX(box)) &&
                   (hypre_IndexX(grid_index) <= hypre_BoxIMaxX(box)) &&
                   (hypre_IndexY(grid_index) >= hypre_BoxIMinY(box)) &&
                   (hypre_IndexY(grid_index) <= hypre_BoxIMaxY(box)) &&
                   (hypre_IndexZ(grid_index) >= hypre_BoxIMinZ(box)) &&
                   (hypre_IndexZ(grid_index) <= hypre_BoxIMaxZ(box))   )
               {
                  for (s = 0; s < num_stencil_indices; s++)
                  {
                     matp = hypre_StructMatrixBoxDataValue(matrix, i,
                                                           stencil_indices[s],
                                                           grid_index);
                    *matp += values[s];
                  }
                  found= true;
               }
            }
         } 

         /* set OffProcAdd for communication */
         if (found)
         {
            hypre_StructMatrixOffProcAdd(matrix)= 1;
         }
         else
         {
            printf("not found- grid_index off the extended matrix grid\n");
         }
      }  /* if ((!found) && (action > 0)) */

   return(ierr);
}

/*--------------------------------------------------------------------------
 * (action > 0): add-to values
 * (action = 0): set values
 * (action =-1): get values and zero out
 * (action <-1): get values
 * should not be called to set a constant-coefficient part of the matrix,
 *   call hypre_StructMatrixSetConstantValues instead
 *--------------------------------------------------------------------------*/

int 
hypre_StructMatrixSetBoxValues( hypre_StructMatrix *matrix,
                                hypre_Box          *value_box,
                                int                 num_stencil_indices,
                                int                *stencil_indices,
                                double             *values,
                                int                 action )
{
   int    ierr = 0;
   MPI_Comm             comm         = hypre_StructMatrixComm(matrix);
   int                 *add_num_ghost= hypre_StructVectorAddNumGhost(matrix);

   hypre_BoxArray      *grid_boxes;
   hypre_Box           *grid_box;
   hypre_BoxArray      *box_array1, *box_array2, *tmp_box_array;
   hypre_BoxArrayArray *box_aarray;
   hypre_Box           *box, *tmp_box, *orig_box;
   hypre_Index          center_index;
   hypre_StructStencil *stencil;
   int                  center_rank;
                   
   hypre_BoxArray      *data_space;
   hypre_Box           *data_box;
   hypre_IndexRef       data_start;
   hypre_Index          data_stride;
   int                  datai;
   double              *datap;
   int                  constant_coefficient;
                   
   hypre_Box           *dval_box;
   hypre_Index          dval_start;
   hypre_Index          dval_stride;
   int                  dvali;
                   
   hypre_Index          loop_size;
                   
   int                  i, j, s, vol_vbox, vol_iboxes, vol_offproc;
   int                  loopi, loopj, loopk;

   int                 nprocs;

   MPI_Comm_size(comm, &nprocs );

   /*-----------------------------------------------------------------------
    * Set up `box_array' by intersecting `box' with the grid boxes
    *-----------------------------------------------------------------------*/

   constant_coefficient = hypre_StructMatrixConstantCoefficient(matrix);

   /* Find the intersecting boxes of the grid with value_box. Record the
      volumes of the intersections for possible off_proc settings. */
   vol_vbox  = hypre_BoxVolume(value_box);
   vol_iboxes= 0;

   grid_boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(matrix));
   box_array1 = hypre_BoxArrayCreate(hypre_BoxArraySize(grid_boxes));
   box = hypre_BoxCreate();
   hypre_ForBoxI(i, grid_boxes)
   {
       grid_box = hypre_BoxArrayBox(grid_boxes, i);
       hypre_IntersectBoxes(value_box, grid_box, box);
       hypre_CopyBox(box, hypre_BoxArrayBox(box_array1, i));
       vol_iboxes+= hypre_BoxVolume(box);
   }

   vol_offproc= 0;
   if ((vol_vbox > vol_iboxes) && (action > 0) && (nprocs > 1)) /* only for addto values */
   {
      box_aarray= hypre_BoxArrayArrayCreate(hypre_BoxArraySize(grid_boxes));

      hypre_ForBoxI(i, grid_boxes)
      {
         tmp_box_array= hypre_BoxArrayCreate(0);
                                                                                           
         /* get ghostlayer boxes */
         orig_box= hypre_BoxArrayBox(grid_boxes, i);
         tmp_box  = hypre_BoxDuplicate(orig_box );
         for (j= 0; j< 3; j++)
         {
            hypre_BoxIMin(tmp_box)[j]-= add_num_ghost[2*j];
            hypre_BoxIMax(tmp_box)[j]+= add_num_ghost[2*j+1];
         }

         /* get ghostlayer boxes */
         hypre_SubtractBoxes(tmp_box, orig_box, tmp_box_array);
         hypre_BoxDestroy(tmp_box);

         box_array2= hypre_BoxArrayArrayBoxArray(box_aarray, i);
         /* intersect the value_box and the ghostlayer boxes */
         hypre_ForBoxI(j, tmp_box_array)
         {
            tmp_box= hypre_BoxArrayBox(tmp_box_array, j);
            hypre_IntersectBoxes(value_box, tmp_box, box);
            hypre_AppendBox(box, box_array2);

            vol_offproc+= hypre_BoxVolume(box);
         }
         hypre_BoxArrayDestroy(tmp_box_array);
      }  /* hypre_ForBoxI(i, grid_boxes) */

      /* if vol_offproc= 0, trying to set values away from ghostlayer */
      if (!vol_offproc)
      {
         hypre_BoxArrayArrayDestroy(box_aarray);
      }
      else
      {
         /* set OffProcAdd for communication, i.e., off-proc values
            must be communicated */
         hypre_StructMatrixOffProcAdd(matrix)= 1;
      }
   }
   hypre_BoxDestroy(box);

   /*-----------------------------------------------------------------------
    * Set the matrix coefficients
    *-----------------------------------------------------------------------*/

   if (box_array1)
   {
      data_space = hypre_StructMatrixDataSpace(matrix);
      hypre_SetIndex(data_stride, 1, 1, 1);

      dval_box = hypre_BoxDuplicate(value_box);
      hypre_BoxIMinD(dval_box, 0) *= num_stencil_indices;
      hypre_BoxIMaxD(dval_box, 0) *= num_stencil_indices;
      hypre_BoxIMaxD(dval_box, 0) += num_stencil_indices - 1;
      hypre_SetIndex(dval_stride, num_stencil_indices, 1, 1);

      hypre_ForBoxI(i, box_array1)
         {
            box      = hypre_BoxArrayBox(box_array1, i);
            data_box = hypre_BoxArrayBox(data_space, i);

            /* if there was an intersection */
            if (box)
            {
               data_start = hypre_BoxIMin(box);
               hypre_CopyIndex(data_start, dval_start);
               hypre_IndexD(dval_start, 0) *= num_stencil_indices;

               if ( constant_coefficient==2 )
               {
                  hypre_SetIndex(center_index, 0, 0, 0);
                  stencil = hypre_StructMatrixStencil(matrix);
                  center_rank = hypre_StructStencilElementRank( stencil, center_index );
               }

               for (s = 0; s < num_stencil_indices; s++)
               {
                  datap = hypre_StructMatrixBoxData(matrix, i,
                                                    stencil_indices[s]);

                  if ( constant_coefficient==1 ||
                       ( constant_coefficient==2 && stencil_indices[s]!=center_rank ))
                     /* datap has only one data point for a given i and s */
                  {
                     ++ierr;  /* should have called SetConstantValues */
                     hypre_BoxGetSize(box, loop_size);

                     if (action > 0)
                     {
                        datai = hypre_CCBoxIndexRank(data_box,data_start);
                        dvali = hypre_BoxIndexRank(dval_box,dval_start);
                        datap[datai] += values[dvali];
                     }
                     else if (action > -1)
                     {
                        datai = hypre_CCBoxIndexRank(data_box,data_start);
                        dvali = hypre_BoxIndexRank(dval_box,dval_start);
                        datap[datai] = values[dvali];
                     }
                     else
                     {
                        datai = hypre_CCBoxIndexRank(data_box,data_start);
                        dvali = hypre_BoxIndexRank(dval_box,dval_start);
                        values[dvali] = datap[datai];
                        if (action == -1)
                        {
                           datap[datai] = 0;
                        }
                     }

                  }
                  else   /* variable coefficient: constant_coefficient==0
                            or diagonal with constant_coefficient==2   */
                  {
                     hypre_BoxGetSize(box, loop_size);

                     if (action > 0)
                     {
                        hypre_BoxLoop2Begin(loop_size,
                                            data_box,data_start,data_stride,datai,
                                            dval_box,dval_start,dval_stride,dvali);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,datai,dvali
#include "hypre_box_smp_forloop.h"
                        hypre_BoxLoop2For(loopi, loopj, loopk, datai, dvali)
                           {
                              datap[datai] += values[dvali];
                           }
                        hypre_BoxLoop2End(datai, dvali);
                     }
                     else if (action > -1)
                     {
                        hypre_BoxLoop2Begin(loop_size,
                                            data_box,data_start,data_stride,datai,
                                            dval_box,dval_start,dval_stride,dvali);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,datai,dvali
#include "hypre_box_smp_forloop.h"
                        hypre_BoxLoop2For(loopi, loopj, loopk, datai, dvali)
                           {
                              datap[datai] = values[dvali];
                           }
                        hypre_BoxLoop2End(datai, dvali);
                     }
                     else
                     {
                        hypre_BoxLoop2Begin(loop_size,
                                            data_box,data_start,data_stride,datai,
                                            dval_box,dval_start,dval_stride,dvali);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,datai,dvali
#include "hypre_box_smp_forloop.h"
                        hypre_BoxLoop2For(loopi, loopj, loopk, datai, dvali)
                           {
                              values[dvali] = datap[datai];
                              if (action == -1)
                              {
                                  datap[datai] = 0;
                              }
                           }
                        hypre_BoxLoop2End(datai, dvali);
                     }
                  }

                  hypre_IndexD(dval_start, 0) ++;
               }
            }
         }

      hypre_BoxDestroy(dval_box);
   }

   hypre_BoxArrayDestroy(box_array1);

   if (vol_offproc) /* only adding values, action > 0 */
   {
      data_space= hypre_StructMatrixDataSpace(matrix);
      hypre_SetIndex(data_stride, 1, 1, 1);

      dval_box = hypre_BoxDuplicate(value_box);
      hypre_BoxIMinD(dval_box, 0) *= num_stencil_indices;
      hypre_BoxIMaxD(dval_box, 0) *= num_stencil_indices;
      hypre_BoxIMaxD(dval_box, 0) += num_stencil_indices - 1;
      hypre_SetIndex(dval_stride, num_stencil_indices, 1, 1);

      hypre_ForBoxI(i, data_space)
      {
         data_box  = hypre_BoxArrayBox(data_space, i);
         box_array2= hypre_BoxArrayArrayBoxArray(box_aarray, i);

         hypre_ForBoxI(j, box_array2)
         {
            box= hypre_BoxArrayBox(box_array2, j);

           /* if there was an intersection */
            if (box)
            {
               data_start = hypre_BoxIMin(box);
               hypre_CopyIndex(data_start, dval_start);
               hypre_IndexD(dval_start, 0) *= num_stencil_indices;

               if ( constant_coefficient==2 )
               {
                  hypre_SetIndex(center_index, 0, 0, 0);
                  stencil= hypre_StructMatrixStencil(matrix);
                  center_rank= 
                       hypre_StructStencilElementRank(stencil, center_index);
               }

               for (s = 0; s < num_stencil_indices; s++)
               {
                  datap = hypre_StructMatrixBoxData(matrix, i,
                                                    stencil_indices[s]);

                  /* variable coefficient: constant_coefficient==0
                     or diagonal with constant_coefficient==2   */
                  if ( constant_coefficient==0 ||
                       ( constant_coefficient==2 && stencil_indices[s]==center_rank ))
                  {
                      hypre_BoxGetSize(box, loop_size);

                      hypre_BoxLoop2Begin(loop_size,
                                          data_box,data_start,data_stride,datai,
                                          dval_box,dval_start,dval_stride,dvali);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,datai,dvali
#include "hypre_box_smp_forloop.h"
                      hypre_BoxLoop2For(loopi, loopj, loopk, datai, dvali)
                      {
                         datap[datai] += values[dvali];
                      }
                      hypre_BoxLoop2End(datai, dvali);

                   }  /* else variable coefficient */
                   hypre_IndexD(dval_start, 0)++;
               }   /* for (s = 0; s < num_stencil_indices; s++) */
            }      /* if (box) */
         }         /* hypre_ForBoxI(j, box_array2) */
      }            /* hypre_ForBoxI(i, data_space) */

      hypre_BoxDestroy(dval_box);
      hypre_BoxArrayArrayDestroy(box_aarray);
   }

   return(ierr);
}


/*--------------------------------------------------------------------------
 * (action > 0): add-to values
 * (action = 0): set values
 * (action =-1): get values and zero out (not implemented, just gets values)
 * (action <-1): get values
 * should be called to set a constant-coefficient part of the matrix
 *--------------------------------------------------------------------------*/

int 
hypre_StructMatrixSetConstantValues( hypre_StructMatrix *matrix,
                                     int             num_stencil_indices,
                                     int            *stencil_indices,
                                     double         *values,
                                     int             action )
{
   int ierr = 0;
   hypre_BoxArray     *boxes;
   hypre_Box          *box;
   hypre_Index        center_index;
   hypre_StructStencil  *stencil;
   int                center_rank;
   int                constant_coefficient;

   double             *matp;

   int                 i, s;

   boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(matrix));
   constant_coefficient = hypre_StructMatrixConstantCoefficient(matrix);

   if ( constant_coefficient==1 )
   {
      hypre_ForBoxI(i, boxes)
         {
            box = hypre_BoxArrayBox(boxes, i);
            if (action > 0)
            {
               for (s = 0; s < num_stencil_indices; s++)
               {
                  matp = hypre_StructMatrixBoxData(matrix, i,
                                                   stencil_indices[s]);
                  *matp += values[s];
               }
            }
            else if (action > -1)
            {
               for (s = 0; s < num_stencil_indices; s++)
               {
                  matp = hypre_StructMatrixBoxData(matrix, i,
                                                   stencil_indices[s]);
                  *matp = values[s];
               }
            }
            else  /* action < 0 */
            {
               for (s = 0; s < num_stencil_indices; s++)
               {
                  matp = hypre_StructMatrixBoxData(matrix, i,
                                                   stencil_indices[s]);
                  values[s] = *matp;
               }
            }
         }
   }
   else if ( constant_coefficient==2 )
   {
      hypre_SetIndex(center_index, 0, 0, 0);
      stencil = hypre_StructMatrixStencil(matrix);
      center_rank = hypre_StructStencilElementRank( stencil, center_index );
      if ( action > 0 )
      {
         for (s = 0; s < num_stencil_indices; s++)
         {
            if ( stencil_indices[s] == center_rank )
            {  /* center (diagonal), like constant_coefficient==0
                  We consider it an error, but do the best we can. */
               ++ierr;
               hypre_ForBoxI(i, boxes)
                  {
                     box = hypre_BoxArrayBox(boxes, i);
                     ierr += hypre_StructMatrixSetBoxValues(
                        matrix, box,
                        num_stencil_indices, stencil_indices,
                        values, action );
                  }
            }
            else
            {  /* non-center, like constant_coefficient==1 */
               matp = hypre_StructMatrixBoxData(matrix, 0,
                                                stencil_indices[s]);
               *matp += values[s];
            }
         }
      }
      else if ( action > -1 )
      {
         for (s = 0; s < num_stencil_indices; s++)
         {
            if ( stencil_indices[s] == center_rank )
            {  /* center (diagonal), like constant_coefficient==0
                  We consider it an error, but do the best we can. */
               ++ierr;
               hypre_ForBoxI(i, boxes)
                  {
                     box = hypre_BoxArrayBox(boxes, i);
                     ierr += hypre_StructMatrixSetBoxValues(
                        matrix, box,
                        num_stencil_indices, stencil_indices,
                        values, action );
                  }
            }
            else
            {  /* non-center, like constant_coefficient==1 */
               matp = hypre_StructMatrixBoxData(matrix, 0,
                                                stencil_indices[s]);
               *matp += values[s];
            }
         }
      }
      else  /* action<0 */
      {
         for (s = 0; s < num_stencil_indices; s++)
         {
            if ( stencil_indices[s] == center_rank )
            {  /* center (diagonal), like constant_coefficient==0
                  We consider it an error, but do the best we can. */
               ++ierr;
               hypre_ForBoxI(i, boxes)
                  {
                     box = hypre_BoxArrayBox(boxes, i);
                     ierr += hypre_StructMatrixSetBoxValues(
                        matrix, box,
                        num_stencil_indices, stencil_indices,
                        values, -1);
                  }
            }
            else
            {  /* non-center, like constant_coefficient==1 */
               matp = hypre_StructMatrixBoxData(matrix, 0,
                                                stencil_indices[s]);
               values[s] = *matp;
            }
         }
      }
   }
   else /* constant_coefficient==0 */
   {
      /* We consider this an error, but do the best we can. */
      ++ierr;
      hypre_ForBoxI(i, boxes)
         {
            box = hypre_BoxArrayBox(boxes, i);
            ierr += hypre_StructMatrixSetBoxValues(
               matrix, box,
               num_stencil_indices, stencil_indices,
               values, action );
         }
   }
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatrixAssemble:
 * Before assembling the matrix, all matrix values added from off_procs
 * are communicated to the correct proc. However, note that since
 * the comm_pkg is created from an "inverted" comm_info derived from the
 * matrix, not all the communicated data is valid (i.e., we did not mark
 * which values were actually set off_proc). Some of them are invalid
 * communicated data. These data values are zero or one (centre). The
 * centre will be checked against 1 (not the best solution).
 *--------------------------------------------------------------------------*/

int 
hypre_StructMatrixAssemble( hypre_StructMatrix *matrix )
{
   int                    ierr = 0;

   int                   *num_ghost = hypre_StructMatrixNumGhost(matrix);

   int                    comm_num_values, mat_num_values, constant_coefficient;
   int                    stencil_size;
   hypre_StructStencil   *stencil;

   hypre_CommInfo        *comm_info;
   hypre_CommPkg         *comm_pkg;

   hypre_CommHandle      *comm_handle;
   int                    data_initial_offset = 0;
   double                *matrix_data = hypre_StructMatrixData(matrix);
   double                *matrix_data_comm = matrix_data;

   int                    sum_OffProcAdd;
   int                    OffProcAdd= hypre_StructMatrixOffProcAdd(matrix);

   hypre_Box             *data_box;
   hypre_Box             *box;
   hypre_Index            loop_size;
   hypre_IndexRef         start;
   hypre_Index            unit_stride;

   int                    i, j;
   int                    loopi, loopj, loopk;

   /* add_values may be off-proc. Communication needed, which is triggered
      if one of the OffProcAdd is 1 */
   sum_OffProcAdd= 0;
   MPI_Allreduce(&OffProcAdd, &sum_OffProcAdd, 1, MPI_INT, MPI_SUM,
                  hypre_StructMatrixComm(matrix));

   /* If matrix_data has an initial segment which is not mesh-based,
      it will not need to be communicated between processors, so
      matrix_data_comm will be set to point to the mesh-based part
      of the data     */

   /*-----------------------------------------------------------------------
    * If the CommPkg has not been set up, set it up
    *
    * The matrix data array is assumed to have two segments - an initial
    * segment of data constant over all space, followed by a segment with
    * comm_num_values matrix entries for each mesh element.  The mesh-dependent
    * data is, of course, the only part relevent to communications.
    * For constant_coefficient==0, all the data is mesh-dependent.
    * For constant_coefficient==1, all  data is constant.
    * For constant_coefficient==2, both segments are non-null.
    *-----------------------------------------------------------------------*/

   constant_coefficient = hypre_StructMatrixConstantCoefficient( matrix );

   mat_num_values = hypre_StructMatrixNumValues(matrix);

   if ( constant_coefficient==0 ) 
   {
       comm_num_values = mat_num_values;
   }    
   else if ( constant_coefficient==1 ) 
   {
       comm_num_values = 0;
   }
   else /* constant_coefficient==2 */
   {
      comm_num_values = 1;
      stencil = hypre_StructMatrixStencil(matrix);
      stencil_size  = hypre_StructStencilSize(stencil);
      data_initial_offset = stencil_size;
      matrix_data_comm = &( matrix_data[data_initial_offset] );
   }

   comm_pkg = hypre_StructMatrixCommPkg(matrix);

   if (!comm_pkg)
   {
      hypre_CreateCommInfoFromNumGhost(hypre_StructMatrixGrid(matrix),
                                       num_ghost, &comm_info);
      hypre_CommPkgCreate(comm_info,
                          hypre_StructMatrixDataSpace(matrix),
                          hypre_StructMatrixDataSpace(matrix),
                          comm_num_values,
                          hypre_StructMatrixComm(matrix), &comm_pkg);

      hypre_StructMatrixCommPkg(matrix) = comm_pkg;

      /* inverse communication pattern if OffProcAdd values.
         Note that we cannot use the comm_info above because the 
         add_num_ghost is generally different from num_ghost (add_num_ghost
         depends on the variable type. */
      if (sum_OffProcAdd)
      {
         /* since the off_proc add_values are located on the ghostlayer, we
            need "inverse" communication. */
          hypre_CommInfo        *add_comm_info;
          hypre_CommInfo        *inv_comm_info;
          hypre_CommPkg         *inv_comm_pkg;
          int                   *add_num_ghost= 
                                     hypre_StructMatrixAddNumGhost(matrix);

          hypre_BoxArrayArray   *send_boxes;
          hypre_BoxArrayArray   *recv_boxes;
          int                  **send_procs;
          int                  **recv_procs;
          int                  **send_rboxnums;
          int                  **recv_rboxnums;
          hypre_BoxArrayArray   *send_rboxes;
          hypre_BoxArray        *box_array, *recv_array;

          double                *data;
          hypre_CommHandle      *inv_comm_handle;

          double                *data_matrix;
          double                *comm_data;
          int                    center_rank;

          int                    s, xi;

          hypre_CreateCommInfoFromNumGhost(hypre_StructMatrixGrid(matrix),
                                           add_num_ghost, &add_comm_info);
         /* inverse CommInfo achieved by switching send & recv structures of
            add_comm_info */
          send_boxes= 
               hypre_BoxArrayArrayDuplicate(hypre_CommInfoRecvBoxes(add_comm_info));
          recv_boxes= 
               hypre_BoxArrayArrayDuplicate(hypre_CommInfoSendBoxes(add_comm_info));
          send_rboxes= hypre_BoxArrayArrayDuplicate(send_boxes);

          send_procs= hypre_CTAlloc(int *, hypre_BoxArrayArraySize(send_boxes));
          recv_procs= hypre_CTAlloc(int *, hypre_BoxArrayArraySize(recv_boxes));
          send_rboxnums= hypre_CTAlloc(int *, hypre_BoxArrayArraySize(send_boxes));
          recv_rboxnums= hypre_CTAlloc(int *, hypre_BoxArrayArraySize(recv_boxes));

          hypre_ForBoxArrayI(i, send_boxes)
          {
             box_array= hypre_BoxArrayArrayBoxArray(send_boxes, i);
             send_procs[i]= hypre_CTAlloc(int, hypre_BoxArraySize(box_array));
             memcpy(send_procs[i], hypre_CommInfoRecvProcesses(add_comm_info)[i],
                    hypre_BoxArraySize(box_array)*sizeof(int));

             send_rboxnums[i]= hypre_CTAlloc(int, hypre_BoxArraySize(box_array));
             memcpy(send_rboxnums[i], hypre_CommInfoRecvRBoxnums(add_comm_info)[i],
                    hypre_BoxArraySize(box_array)*sizeof(int));
          }

          hypre_ForBoxArrayI(i, recv_boxes)
          {
             box_array= hypre_BoxArrayArrayBoxArray(recv_boxes, i);
             recv_procs[i]= hypre_CTAlloc(int, hypre_BoxArraySize(box_array));
             memcpy(recv_procs[i], hypre_CommInfoSendProcesses(add_comm_info)[i],
                    hypre_BoxArraySize(box_array)*sizeof(int));

             recv_rboxnums[i]= hypre_CTAlloc(int, hypre_BoxArraySize(box_array));
             memcpy(recv_rboxnums[i], hypre_CommInfoSendRBoxnums(add_comm_info)[i],
                    hypre_BoxArraySize(box_array)*sizeof(int));
          }

          hypre_CommInfoCreate(send_boxes, recv_boxes, send_procs, recv_procs,
                               send_rboxnums, recv_rboxnums, send_rboxes,
                              &inv_comm_info);

          hypre_CommPkgCreate(inv_comm_info,
                              hypre_StructMatrixDataSpace(matrix),
                              hypre_StructMatrixDataSpace(matrix),
                              comm_num_values,
                              hypre_StructMatrixComm(matrix),
                              &inv_comm_pkg);

          /* communicate the add value entries */
          data= hypre_CTAlloc(double, hypre_StructMatrixDataSize(matrix));
          hypre_InitializeCommunication(inv_comm_pkg,
                                        hypre_StructMatrixData(matrix),
                                        data,
                                       &inv_comm_handle);
          hypre_FinalizeCommunication(inv_comm_handle);

         /* this proc will recved data in it's send_boxes of add_comm_info, or
            equivalently, in the recv_boxes of inv_comm_info. Since inv_comm_info
            has already been destroyed, we use the send_boxes of add_comm_info */
          stencil = hypre_StructMatrixStencil(matrix);
          stencil_size  = hypre_StructStencilSize(stencil);
          hypre_SetIndex(unit_stride, 0, 0, 0);
          center_rank = hypre_StructStencilElementRank(stencil, unit_stride);
          hypre_SetIndex(unit_stride, 1, 1, 1);

          box_array= hypre_StructGridBoxes(hypre_StructMatrixGrid(matrix));
          hypre_ForBoxI(i, box_array)
          {
             recv_array=
                hypre_BoxArrayArrayBoxArray(hypre_CommInfoSendBoxes(add_comm_info), i);

             data_box= hypre_BoxArrayBox(hypre_StructMatrixDataSpace(matrix), i);
             for (s= 0; s< stencil_size; s++)
             {
                data_matrix= hypre_StructMatrixBoxData(matrix, i, s);
                comm_data=(data + hypre_StructMatrixDataIndices(matrix)[i][s]);

                hypre_ForBoxI(j, recv_array)
                {
                   box  = hypre_BoxArrayBox(recv_array, j);
                   start= hypre_BoxIMin(box);
                   hypre_BoxGetSize(box, loop_size);
                  
                  /* only adding offproc values. */
                   if (s != center_rank)
                   {
                       hypre_BoxLoop1Begin(loop_size,
                                           data_box, start, unit_stride, xi)
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,xi
#include "hypre_box_smp_forloop.h"
                       hypre_BoxLoop1For(loopi, loopj, loopk, xi)
                       {
                           data_matrix[xi] += comm_data[xi];
                       }
                       hypre_BoxLoop1End(xi);
                   }

                   else
                   {
                       hypre_BoxLoop1Begin(loop_size,
                                           data_box, start, unit_stride, xi)
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,xi
#include "hypre_box_smp_forloop.h"
                       hypre_BoxLoop1For(loopi, loopj, loopk, xi)
                       {
                          if (comm_data[xi] != 1.0)
                          {
                              data_matrix[xi] += comm_data[xi];
                          }
                       }
                       hypre_BoxLoop1End(xi);
                   }

                }  /* hypre_ForBoxI(j, recv_array) */
             }     /* for (s= 0; s< stencil_size; s++) */
          }        /* hypre_ForBoxI(i, box_array) */

          hypre_TFree(data);
          hypre_CommPkgDestroy(inv_comm_pkg);
          hypre_CommInfoDestroy(add_comm_info);
      }
   }

   /*-----------------------------------------------------------------------
    * Update the ghost data
    * This takes care of the communication needs of all known functions
    * referencing the matrix.
    *
    * At present this is the only place where matrix data gets communicated.
    * However, comm_pkg is kept as long as the matrix is, in case some
    * future version hypre has a use for it - e.g. if the user replaces
    * a matrix with a very similar one, we may not want to recompute comm_pkg.
    *-----------------------------------------------------------------------*/

   if ( constant_coefficient!=1 )
   {
      hypre_InitializeCommunication( comm_pkg,
                                     matrix_data_comm,
                                     matrix_data_comm,
                                     &comm_handle );
      hypre_FinalizeCommunication( comm_handle );
   }

   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_StructMatrixSetNumGhost
 *--------------------------------------------------------------------------*/

int
hypre_StructMatrixSetNumGhost( hypre_StructMatrix *matrix,
                               int                *num_ghost )
{
   int  ierr = 0;
   int  i;

   for (i = 0; i < 6; i++)
      hypre_StructMatrixNumGhost(matrix)[i] = num_ghost[i];

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatrixSetConstantCoefficient
 * deprecated in user interface, in favor of SetConstantEntries.
 * left here for internal use
 *--------------------------------------------------------------------------*/

int
hypre_StructMatrixSetConstantCoefficient( hypre_StructMatrix *matrix,
                                          int                constant_coefficient )
{
   int  ierr = 0;

   hypre_StructMatrixConstantCoefficient(matrix) = constant_coefficient;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatrixSetConstantEntries
 * - nentries is the number of array entries
 * - Each int entries[i] is an index into the shape array of the stencil of the
 * matrix
 * In the present version, only three possibilites are recognized:
 * - no entries constant                 (constant_coefficient==0)
 * - all entries constant                (constant_coefficient==1)
 * - all but the diagonal entry constant (constant_coefficient==2)
 * If something else is attempted, this function will return a nonzero error.
 * In the present version, if this function is called more than once, only
 * the last call will take effect.
 *--------------------------------------------------------------------------*/


int  hypre_StructMatrixSetConstantEntries( hypre_StructMatrix *matrix,
                                           int                 nentries,
                                           int                *entries )
{
   /* We make an array offdconst corresponding to the stencil's shape array,
      and use "entries" to fill it with flags - 1 for constant, 0 otherwise.
      By counting the nonzeros in offdconst, and by checking whether its
      diagonal entry is nonzero, we can distinguish among the three
      presently legal values of constant_coefficient, and detect input errors.
      We do not need to treat duplicates in "entries" as an error condition.
   */
   int ierr = 0;
   hypre_StructStencil *stencil = hypre_StructMatrixUserStencil(matrix);
   /* ... Stencil doesn't exist yet */
   int stencil_size  = hypre_StructStencilSize(stencil);
   int *offdconst = hypre_CTAlloc(int, stencil_size);
   /* ... note: CTAlloc initializes to 0 (normally it works by calling calloc) */
   int nconst = 0;
   int constant_coefficient, diag_rank;
   hypre_Index diag_index;
   int i, j;

   for ( i=0; i<nentries; ++i )
   {
      offdconst[ entries[i] ] = 1;
   }
   for ( j=0; j<stencil_size; ++j )
   {
      nconst += offdconst[j];
   }
   if ( nconst<=0 ) constant_coefficient=0;
   else if ( nconst>=stencil_size ) constant_coefficient=1;
   else
   {
      hypre_SetIndex(diag_index, 0, 0, 0);
      diag_rank = hypre_StructStencilElementRank( stencil, diag_index );
      if ( offdconst[diag_rank]==0 )
      {
         constant_coefficient=2;
         if ( nconst!=(stencil_size-1) ) ++ierr;
      }
      else
      {
         constant_coefficient=0;
         ++ierr;
      }
   }

   ierr += hypre_StructMatrixSetConstantCoefficient( matrix, constant_coefficient );

   hypre_TFree(offdconst);
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatrixPrint
 *--------------------------------------------------------------------------*/

int
hypre_StructMatrixPrint( const char         *filename,
                         hypre_StructMatrix *matrix,
                         int                 all      )
{
   int                   ierr = 0;

   FILE                 *file;
   char                  new_filename[255];

   hypre_StructGrid     *grid;
   hypre_BoxArray       *boxes;

   hypre_StructStencil  *stencil;
   hypre_Index          *stencil_shape;
   int                   stencil_size;
   hypre_Index           center_index;

   int                   num_values;

   hypre_BoxArray       *data_space;

   int                  *symm_elements;
                   
   int                   i, j;
   int                   constant_coefficient;
   int                   center_rank;
   int                   myid;

   constant_coefficient = hypre_StructMatrixConstantCoefficient(matrix);

   /*----------------------------------------
    * Open file
    *----------------------------------------*/

#ifdef HYPRE_USE_PTHREADS 
#if MPI_Comm_rank == hypre_thread_MPI_Comm_rank
#undef MPI_Comm_rank
#endif
#endif

   MPI_Comm_rank(hypre_StructMatrixComm(matrix), &myid);

   sprintf(new_filename, "%s.%05d", filename, myid);
 
   if ((file = fopen(new_filename, "w")) == NULL)
   {
      printf("Error: can't open output file %s\n", new_filename);
      exit(1);
   }

   /*----------------------------------------
    * Print header info
    *----------------------------------------*/

   fprintf(file, "StructMatrix\n");

   fprintf(file, "\nSymmetric: %d\n", hypre_StructMatrixSymmetric(matrix));
   fprintf(file, "\nConstantCoefficient: %d\n", hypre_StructMatrixConstantCoefficient(matrix));

   /* print grid info */
   fprintf(file, "\nGrid:\n");
   grid = hypre_StructMatrixGrid(matrix);
   hypre_StructGridPrint(file, grid);

   /* print stencil info */
   fprintf(file, "\nStencil:\n");
   stencil = hypre_StructMatrixStencil(matrix);
   stencil_shape = hypre_StructStencilShape(stencil);

   num_values = hypre_StructMatrixNumValues(matrix);
   symm_elements = hypre_StructMatrixSymmElements(matrix);
   fprintf(file, "%d\n", num_values);
   stencil_size = hypre_StructStencilSize(stencil);
   j = 0;
   for (i=0; i<stencil_size; i++)
   {
      if (symm_elements[i] < 0)
      {
         fprintf(file, "%d: %d %d %d\n", j++,
                 hypre_IndexX(stencil_shape[i]),
                 hypre_IndexY(stencil_shape[i]),
                 hypre_IndexZ(stencil_shape[i]));
      }
   }

   /*----------------------------------------
    * Print data
    *----------------------------------------*/

   data_space = hypre_StructMatrixDataSpace(matrix);
 
   if (all)
      boxes = data_space;
   else
      boxes = hypre_StructGridBoxes(grid);
 
   fprintf(file, "\nData:\n");
   if ( constant_coefficient==1 )
   {
      hypre_PrintCCBoxArrayData(file, boxes, data_space, num_values,
                              hypre_StructMatrixData(matrix));
   }
   else if ( constant_coefficient==2 )
   {
      hypre_SetIndex(center_index, 0, 0, 0);
      center_rank = hypre_StructStencilElementRank( stencil, center_index );

      hypre_PrintCCVDBoxArrayData(file, boxes, data_space, num_values,
                                  center_rank, stencil_size, symm_elements,
                                  hypre_StructMatrixData(matrix));
   }
   else
   {
      hypre_PrintBoxArrayData(file, boxes, data_space, num_values,
                              hypre_StructMatrixData(matrix));
   }

   /*----------------------------------------
    * Close file
    *----------------------------------------*/
 
   fflush(file);
   fclose(file);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatrixMigrate
 *--------------------------------------------------------------------------*/

int 
hypre_StructMatrixMigrate( hypre_StructMatrix *from_matrix,
                           hypre_StructMatrix *to_matrix   )
{
   hypre_CommInfo        *comm_info;
   hypre_CommPkg         *comm_pkg;
   hypre_CommHandle      *comm_handle;

   int                    ierr = 0;
   int                    constant_coefficient, comm_num_values;
   int                    stencil_size, mat_num_values;
   hypre_StructStencil   *stencil;
   int                    data_initial_offset = 0;
   double                *matrix_data_from = hypre_StructMatrixData(from_matrix);
   double                *matrix_data_to = hypre_StructMatrixData(to_matrix);
   double                *matrix_data_comm_from = matrix_data_from;
   double                *matrix_data_comm_to = matrix_data_to;

   /*------------------------------------------------------
    * Set up hypre_CommPkg
    *------------------------------------------------------*/

   constant_coefficient = hypre_StructMatrixConstantCoefficient( from_matrix );
   hypre_assert( constant_coefficient == hypre_StructMatrixConstantCoefficient( to_matrix ) );

   mat_num_values = hypre_StructMatrixNumValues(from_matrix);
   hypre_assert( mat_num_values = hypre_StructMatrixNumValues(to_matrix) );

   if ( constant_coefficient==0 ) 
   {
       comm_num_values = mat_num_values;
   }
   else if ( constant_coefficient==1 ) 
   {
       comm_num_values = 0;
   }
   else /* constant_coefficient==2 */ 
   {
      comm_num_values = 1;
      stencil = hypre_StructMatrixStencil(from_matrix);
      stencil_size = hypre_StructStencilSize(stencil);
      hypre_assert(stencil_size ==
             hypre_StructStencilSize( hypre_StructMatrixStencil(to_matrix) ) );
      data_initial_offset = stencil_size;
      matrix_data_comm_from = &( matrix_data_from[data_initial_offset] );
      matrix_data_comm_to = &( matrix_data_to[data_initial_offset] );
   }

   hypre_CreateCommInfoFromGrids(hypre_StructMatrixGrid(from_matrix),
                                 hypre_StructMatrixGrid(to_matrix),
                                 &comm_info);
   hypre_CommPkgCreate(comm_info,
                       hypre_StructMatrixDataSpace(from_matrix),
                       hypre_StructMatrixDataSpace(to_matrix),
                       comm_num_values,
                       hypre_StructMatrixComm(from_matrix), &comm_pkg);
   /* is this correct for periodic? */

   /*-----------------------------------------------------------------------
    * Migrate the matrix data
    *-----------------------------------------------------------------------*/
 
   if ( constant_coefficient!=1 )
   {
      hypre_InitializeCommunication( comm_pkg,
                                     matrix_data_comm_from,
                                     matrix_data_comm_to,
                                     &comm_handle );
      hypre_FinalizeCommunication( comm_handle );
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatrixRead
 *--------------------------------------------------------------------------*/

hypre_StructMatrix *
hypre_StructMatrixRead( MPI_Comm    comm,
                        const char *filename,
                        int        *num_ghost )
{
   FILE                 *file;
   char                  new_filename[255];
                      
   hypre_StructMatrix   *matrix;

   hypre_StructGrid     *grid;
   hypre_BoxArray       *boxes;
   int                   dim;

   hypre_StructStencil  *stencil;
   hypre_Index          *stencil_shape;
   int                   stencil_size, real_stencil_size;

   int                   num_values;

   hypre_BoxArray       *data_space;

   int                   symmetric;
   int                   constant_coefficient;
                       
   int                   i, idummy;
                       
   int                   myid;

   /*----------------------------------------
    * Open file
    *----------------------------------------*/

#ifdef HYPRE_USE_PTHREADS
#if MPI_Comm_rank == hypre_thread_MPI_Comm_rank
#undef MPI_Comm_rank
#endif
#endif

   MPI_Comm_rank(comm, &myid );

   sprintf(new_filename, "%s.%05d", filename, myid);
 
   if ((file = fopen(new_filename, "r")) == NULL)
   {
      printf("Error: can't open output file %s\n", new_filename);
      exit(1);
   }

   /*----------------------------------------
    * Read header info
    *----------------------------------------*/

   fscanf(file, "StructMatrix\n");

   fscanf(file, "\nSymmetric: %d\n", &symmetric);
   fscanf(file, "\nConstantCoefficient: %d\n", &constant_coefficient);

   /* read grid info */
   fscanf(file, "\nGrid:\n");
   hypre_StructGridRead(comm,file,&grid);

   /* read stencil info */
   fscanf(file, "\nStencil:\n");
   dim = hypre_StructGridDim(grid);
   fscanf(file, "%d\n", &stencil_size);
   if (symmetric) { real_stencil_size = 2*stencil_size-1; }
   else { real_stencil_size = stencil_size; }
   /* ... real_stencil_size is the stencil size of the matrix after it's fixed up
      by the call (if any) of hypre_StructStencilSymmetrize from
      hypre_StructMatrixInitializeShell.*/
   stencil_shape = hypre_CTAlloc(hypre_Index, stencil_size);
   for (i = 0; i < stencil_size; i++)
   {
      fscanf(file, "%d: %d %d %d\n", &idummy,
             &hypre_IndexX(stencil_shape[i]),
             &hypre_IndexY(stencil_shape[i]),
             &hypre_IndexZ(stencil_shape[i]));
   }
   stencil = hypre_StructStencilCreate(dim, stencil_size, stencil_shape);

   /*----------------------------------------
    * Initialize the matrix
    *----------------------------------------*/

   matrix = hypre_StructMatrixCreate(comm, grid, stencil);
   hypre_StructMatrixSymmetric(matrix) = symmetric;
   hypre_StructMatrixConstantCoefficient(matrix) = constant_coefficient;
   hypre_StructMatrixSetNumGhost(matrix, num_ghost);
   hypre_StructMatrixInitialize(matrix);

   /*----------------------------------------
    * Read data
    *----------------------------------------*/

   boxes      = hypre_StructGridBoxes(grid);
   data_space = hypre_StructMatrixDataSpace(matrix);
   num_values = hypre_StructMatrixNumValues(matrix);
 
   fscanf(file, "\nData:\n");
   if ( constant_coefficient==0 )
   {
      hypre_ReadBoxArrayData(file, boxes, data_space, num_values,
                             hypre_StructMatrixData(matrix));
   }
   else
   {
      hypre_assert( constant_coefficient<=2 );
      hypre_ReadBoxArrayData_CC( file, boxes, data_space,
                                 stencil_size, real_stencil_size,
                                 constant_coefficient,
                                 hypre_StructMatrixData(matrix));
   }

   /*----------------------------------------
    * Assemble the matrix
    *----------------------------------------*/

   hypre_StructMatrixAssemble(matrix);

   /*----------------------------------------
    * Close file
    *----------------------------------------*/
 
   fclose(file);

   return matrix;
}

