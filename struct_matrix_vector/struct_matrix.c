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
 * Member functions for zzz_StructMatrix class.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * zzz_NewStructMatrix
 *--------------------------------------------------------------------------*/

zzz_StructMatrix *
zzz_NewStructMatrix( zzz_StructGrid    *grid,
                     zzz_StructStencil *user_stencil )
{
   zzz_StructMatrix  *matrix;

   int                i;

   matrix = ctalloc(zzz_StructMatrix, 1);

   zzz_StructMatrixGrid(matrix)        = grid;
   zzz_StructMatrixUserStencil(matrix) = user_stencil;

   /* set defaults */
   zzz_StructMatrixSymmetric(matrix) = 0;
   for (i = 0; i < 6; i++)
      zzz_StructMatrixNumGhost(matrix)[i] = 0;

   return matrix;
}

/*--------------------------------------------------------------------------
 * zzz_FreeStructMatrix
 *--------------------------------------------------------------------------*/

int 
zzz_FreeStructMatrix( zzz_StructMatrix *matrix )
{
   int  ierr;

   int  i;

   if (matrix)
   {
      zzz_FreeCommPkg(zzz_StructMatrixCommPkg(matrix));

      zzz_ForBoxI(i, zzz_StructMatrixDataSpace(matrix))
         tfree(zzz_StructMatrixDataIndices(matrix)[i]);
      tfree(zzz_StructMatrixDataIndices(matrix));
      tfree(zzz_StructMatrixData(matrix));

      zzz_FreeBoxArray(zzz_StructMatrixDataSpace(matrix));

      if (zzz_StructMatrixSymmetric(matrix))
      {
         tfree(zzz_StructMatrixSymmCoeff(matrix));
         zzz_FreeStructStencil(zzz_StructMatrixStencil(matrix));
      }

      tfree(matrix);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_InitializeStructMatrixShell
 *--------------------------------------------------------------------------*/

int 
zzz_InitializeStructMatrixShell( zzz_StructMatrix *matrix )
{
   int    ierr;

   zzz_StructGrid     *grid;

   zzz_StructStencil  *stencil;
   zzz_Index         **stencil_shape;
   int                 stencil_size;
   zzz_StructStencil  *user_stencil;
   zzz_Index         **user_stencil_shape;
   int                 user_stencil_size;
   int                 num_values;
   int                *symm_coeff;
   int                 no_symmetric_stencil_element;

   int                *num_ghost;
 
   zzz_BoxArray       *data_space;
   zzz_BoxArray       *boxes;
   zzz_Box            *box;
   zzz_Box            *data_box;

   int               **data_indices;
   int                 data_size;
   int                 data_box_volume;

   int                 i, j, d;
 
   grid = zzz_StructMatrixGrid(matrix);

   /*-----------------------------------------------------------------------
    * Set up stencil:
    *
    *------------------
    * Non-symmetric case:
    *    Just set pointer to user's stencil, `user_stencil'.
    *
    *------------------
    * Symmetric case:
    *    Copy the user's stencil elements into the new stencil, and
    *    create symmetric stencil elements if needed.
    *
    *    Also set up an array called `symm_coeff'.  A non-zero value
    *    of `symm_coeff[i]' indicates that the `i'th stencil element
    *    is a "symmetric element".  That is, the data associated with
    *    stencil element `i' is not explicitely stored, but is instead
    *    stored as the transpose coefficient at a neighboring grid point.
    *    The value of `symm_coeff[i]' is also the index of the transpose
    *    stencil element.
    *-----------------------------------------------------------------------*/

   user_stencil = zzz_StructMatrixUserStencil(matrix);

   /* non-symmetric case */
   if (!zzz_StructMatrixSymmetric(matrix))
   {
      stencil = user_stencil;
      stencil_size  = zzz_StructStencilSize(stencil);
      stencil_shape = zzz_StructStencilShape(stencil);

      num_values = stencil_size;
   }

   /* symmetric case */
   else
   {
      user_stencil_shape = zzz_StructStencilShape(user_stencil);
      user_stencil_size = zzz_StructStencilSize(user_stencil);

      /* copy user's stencil elements into `stencil_shape' */
      stencil_shape = ctalloc(zzz_Index *, 2*user_stencil_size);
      for (i = 0; i < user_stencil_size; i++)
      {
         stencil_shape[i] = zzz_NewIndex();
         zzz_CopyIndex(user_stencil_shape[i], stencil_shape[i]);
      }

      /* create symmetric stencil elements and `symm_coeff' */
      symm_coeff = ctalloc(int, 2*user_stencil_size);
      stencil_size = user_stencil_size;
      for (i = 0; i < user_stencil_size; i++)
      {
	 if (!symm_coeff[i])
	 {
            /* note: start at i to handle "center" element correctly */
            no_symmetric_stencil_element = 1;
            for (j = i; j < user_stencil_size; j++)
            {
	       if ( (zzz_IndexX(stencil_shape[j]) ==
                     -zzz_IndexX(stencil_shape[i])  ) &&
                    (zzz_IndexY(stencil_shape[j]) ==
                     -zzz_IndexY(stencil_shape[i])  ) &&
                    (zzz_IndexZ(stencil_shape[j]) ==
                     -zzz_IndexZ(stencil_shape[i])  )   )
	       {
                  /* only "off-center" elements have symmetric entries */
                  if (i != j)
                     symm_coeff[j] = i;
                  no_symmetric_stencil_element = 0;
	       }
            }

            if (no_symmetric_stencil_element)
            {
               /* add symmetric stencil element to `stencil' */
               stencil_shape[stencil_size] = zzz_NewIndex();
               for (d = 0; d < 3; d++)
               {
                  zzz_IndexD(stencil_shape[stencil_size], d) =
                     -zzz_IndexD(stencil_shape[i], d);
               }
	       
               symm_coeff[stencil_size] = i;
               stencil_size++;
	    }
	 }
      }

      stencil = zzz_NewStructStencil(zzz_StructStencilDim(user_stencil),
                                     stencil_size, stencil_shape);
      zzz_StructMatrixSymmCoeff(matrix) = symm_coeff;

      num_values = (stencil_size + 1) / 2;
   }

   zzz_StructMatrixStencil(matrix) = stencil;
   zzz_StructMatrixNumValues(matrix) = num_values;

   /*-----------------------------------------------------------------------
    * Set ghost-layer size for symmetric storage
    *-----------------------------------------------------------------------*/

   num_ghost = zzz_StructMatrixNumGhost(matrix);

   if (zzz_StructMatrixSymmetric(matrix))
   {
      for (i = 0; i < stencil_size; i++)
      {
         if (symm_coeff[i])
         {
            j = 0;
            for (d = 0; d < 3; d++)
            {
               num_ghost[j] =
                  max(num_ghost[  j], -zzz_IndexD(stencil_shape[i], d));
               num_ghost[j+1] =
                  max(num_ghost[j+1],  zzz_IndexD(stencil_shape[i], d));
               j += 2;
            }
         }
      }
   }

   /*-----------------------------------------------------------------------
    * Set up data_space
    *-----------------------------------------------------------------------*/

   boxes = zzz_StructGridBoxes(grid);
   data_space = zzz_NewBoxArray();

   zzz_ForBoxI(i, boxes)
   {
      box = zzz_BoxArrayBox(boxes, i);

      data_box = zzz_DuplicateBox(box);
      if (zzz_BoxVolume(data_box))
      {
         for (d = 0; d < 3; d++)
         {
            zzz_BoxIMinD(data_box, d) -= num_ghost[2*d];
            zzz_BoxIMaxD(data_box, d) += num_ghost[2*d + 1];
         }
      }

      zzz_AppendBox(data_box, data_space);
   }

   zzz_StructMatrixDataSpace(matrix) = data_space;

   /*-----------------------------------------------------------------------
    * Set up data_indices array and data-size
    *-----------------------------------------------------------------------*/

   data_indices = ctalloc(int *, zzz_BoxArraySize(data_space));

   data_size = 0;
   zzz_ForBoxI(i, data_space)
   {
      data_box = zzz_BoxArrayBox(data_space, i);
      data_box_volume  = zzz_BoxVolume(data_box);

      data_indices[i] = ctalloc(int, stencil_size);

      /* non-symmetric case */
      if (!zzz_StructMatrixSymmetric(matrix))
      {
         for (j = 0; j < stencil_size; j++)
	 {
	    data_indices[i][j] = data_size;
	    data_size += data_box_volume;
	 }
      }

      /* symmetric case */
      else
      {
         /* set pointers for "stored" coefficients */
         for (j = 0; j < stencil_size; j++)
         {
            if (!symm_coeff[j])
            {
               data_indices[i][j] = data_size;
               data_size += data_box_volume;
            }
         }

         /* set pointers for "symmetric" coefficients */
         for (j = 0; j < stencil_size; j++)
         {
            if (symm_coeff[j])
            {
               data_indices[i][j] = data_indices[i][symm_coeff[j]] +
                  zzz_BoxOffsetDistance(data_box, stencil_shape[j]);
            }
         }
      }
   }

   zzz_StructMatrixDataIndices(matrix) = data_indices;
   zzz_StructMatrixDataSize(matrix)    = data_size;

   /*-----------------------------------------------------------------------
    * Set total number of nonzero coefficients
    *-----------------------------------------------------------------------*/

   zzz_StructMatrixGlobalSize(matrix) =
      zzz_StructGridGlobalSize(grid) * stencil_size;

   /*-----------------------------------------------------------------------
    * Return
    *-----------------------------------------------------------------------*/

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_InitializeStructMatrixData
 *--------------------------------------------------------------------------*/

void
zzz_InitializeStructMatrixData( zzz_StructMatrix *matrix,
                                double           *data   )
{
   zzz_StructMatrixData(matrix) = data;
}

/*--------------------------------------------------------------------------
 * zzz_InitializeStructMatrix
 *--------------------------------------------------------------------------*/

int 
zzz_InitializeStructMatrix( zzz_StructMatrix *matrix )
{
   int    ierr;

   double *data;

   ierr = zzz_InitializeStructMatrixShell(matrix);

   data = ctalloc(double, zzz_StructMatrixDataSize(matrix));
   zzz_InitializeStructMatrixData(matrix, data);

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SetStructMatrixValues
 *--------------------------------------------------------------------------*/

int 
zzz_SetStructMatrixValues( zzz_StructMatrix *matrix,
                           zzz_Index        *grid_index,
                           int               num_stencil_indices,
                           int              *stencil_indices,
                           double           *values              )
{
   int    ierr;

   zzz_BoxArray     *boxes;
   zzz_Box          *box;
   zzz_Index        *imin;
   zzz_Index        *imax;

   double           *matp;

   int               i, s;

   boxes = zzz_StructGridBoxes(zzz_StructMatrixGrid(matrix));

   zzz_ForBoxI(i, boxes)
   {
      box = zzz_BoxArrayBox(boxes, i);
      imin = zzz_BoxIMin(box);
      imax = zzz_BoxIMax(box);

      if ((zzz_IndexX(grid_index) >= zzz_IndexX(imin)) &&
          (zzz_IndexX(grid_index) <= zzz_IndexX(imax)) &&
          (zzz_IndexY(grid_index) >= zzz_IndexY(imin)) &&
          (zzz_IndexY(grid_index) <= zzz_IndexY(imax)) &&
          (zzz_IndexZ(grid_index) >= zzz_IndexZ(imin)) &&
          (zzz_IndexZ(grid_index) <= zzz_IndexZ(imax))   )
      {
         for (s = 0; s < num_stencil_indices; s++)
         {
            matp = zzz_StructMatrixBoxDataValue(matrix, i, stencil_indices[s],
                                                grid_index);
            *matp = values[s];
         }
      }
   }

   return(ierr);
}

/*--------------------------------------------------------------------------
 * zzz_SetStructMatrixBoxValues
 *--------------------------------------------------------------------------*/

int 
zzz_SetStructMatrixBoxValues( zzz_StructMatrix *matrix,
                              zzz_Box          *value_box,
                              int               num_stencil_indices,
                              int              *stencil_indices,
                              double           *values              )
{
   int    ierr;

   zzz_BoxArray     *grid_boxes;
   zzz_Box          *grid_box;
   zzz_BoxArray     *box_array;
   zzz_Box          *box;

   zzz_BoxArray     *data_space;
   zzz_Box          *data_box;
   zzz_Index        *data_start;
   zzz_Index        *data_stride;
   int               datai;
   double           *datap;

   zzz_Box          *dval_box;
   zzz_Index        *dval_start;
   zzz_Index        *dval_stride;
   int               dvali;

   zzz_Index        *index;

   int               i, s;

   /*-----------------------------------------------------------------------
    * Set up `box_array' by intersecting `box' with the grid boxes
    *-----------------------------------------------------------------------*/

   box_array = zzz_NewBoxArray();
   grid_boxes = zzz_StructGridBoxes(zzz_StructMatrixGrid(matrix));
   zzz_ForBoxI(i, grid_boxes)
   {
      grid_box = zzz_BoxArrayBox(grid_boxes, i);
      box = zzz_IntersectBoxes(value_box, grid_box);
      zzz_AppendBox(box, box_array);
   }

   /*-----------------------------------------------------------------------
    * Set the matrix coefficients
    *-----------------------------------------------------------------------*/

   if (box_array)
   {
      index = zzz_NewIndex();

      data_space = zzz_StructMatrixDataSpace(matrix);
      data_stride = zzz_NewIndex();
      zzz_SetIndex(data_stride, 1, 1, 1);

      dval_box = zzz_DuplicateBox(value_box);
      zzz_BoxIMaxD(dval_box, 0) +=
         (num_stencil_indices - 1)*zzz_BoxSizeD(dval_box, 0);
      dval_stride = zzz_NewIndex();
      zzz_SetIndex(dval_stride, num_stencil_indices, 1, 1);
      dval_start = zzz_NewIndex();

      zzz_ForBoxI(i, box_array)
      {
         box      = zzz_BoxArrayBox(box_array, i);
         data_box = zzz_BoxArrayBox(data_space, i);

         /* if there was an intersection */
         if (box)
         {
            data_start = zzz_BoxIMin(box);
            zzz_CopyIndex(data_start, dval_start);

            for (s = 0; s < num_stencil_indices; s++)
            {
               datap = zzz_StructMatrixBoxData(matrix, i, stencil_indices[s]);

               zzz_BoxLoop2(box, index,
                            data_box, data_start, data_stride, datai,
                            dval_box, dval_start, dval_stride, dvali,
                            {
                               datap[datai] = values[dvali];
                            });

               zzz_IndexD(dval_start, 0) ++;
            }
         }
      }

      zzz_FreeBox(dval_box);
      zzz_FreeIndex(dval_start);
      zzz_FreeIndex(dval_stride);
      zzz_FreeIndex(data_stride);
      zzz_FreeIndex(index);
   }

   zzz_FreeBoxArray(box_array);

   return(ierr);
}

/*--------------------------------------------------------------------------
 * zzz_AssembleStructMatrix
 *--------------------------------------------------------------------------*/

int 
zzz_AssembleStructMatrix( zzz_StructMatrix *matrix )
{
   int    ierr;

   zzz_StructStencil   *comm_stencil;
   zzz_Index          **comm_stencil_shape;
   int                 *num_ghost;
   zzz_Box             *box;
   zzz_Index           *imin;
   zzz_Index           *imax;
   zzz_Index           *index;

   zzz_BoxArrayArray   *send_boxes;
   zzz_BoxArrayArray   *recv_boxes;

   zzz_SBoxArrayArray  *send_sboxes;
   zzz_SBoxArrayArray  *recv_sboxes;
   int                **send_box_ranks;
   int                **recv_box_ranks;
   zzz_CommPkg         *comm_pkg;

   int                  i, d;

   zzz_CommHandle      *comm_handle;

   /*-----------------------------------------------------------------------
    * If the CommPkg has not been set up, set it up
    *-----------------------------------------------------------------------*/

   comm_pkg = zzz_StructMatrixCommPkg(matrix);

   if (!comm_pkg)
   {
      /* Set up the stencil describing communications, `comm_stencil' */

      num_ghost = zzz_StructMatrixNumGhost(matrix);
      imin = zzz_NewIndex();
      imax = zzz_NewIndex();
      for (d = 0; d < 3; d++)
      {
         zzz_IndexD(imin, d) = -num_ghost[2*d];
         zzz_IndexD(imax, d) =  num_ghost[2*d + 1];
      }
      box = zzz_NewBox(imin, imax);

      comm_stencil_shape = ctalloc(zzz_Index *, zzz_BoxVolume(box));
      index = zzz_NewIndex();
      i = 0;
      zzz_BoxLoop0(box, index,
                   {
                      comm_stencil_shape[i] = zzz_NewIndex();
                      zzz_CopyIndex(index, comm_stencil_shape[i]);
                      i++;
                   });
      comm_stencil = zzz_NewStructStencil(3, zzz_BoxVolume(box),
                                          comm_stencil_shape);

      zzz_FreeIndex(index);
      zzz_FreeBox(box);

      /* Set up the CommPkg */

      zzz_GetCommInfo(&send_boxes, &recv_boxes,
                      &send_box_ranks, &recv_box_ranks,
                      zzz_StructMatrixGrid(matrix),
                      comm_stencil);

      send_sboxes = zzz_ConvertToSBoxArrayArray(send_boxes);
      recv_sboxes = zzz_ConvertToSBoxArrayArray(recv_boxes);

      comm_pkg = zzz_NewCommPkg(send_sboxes, recv_sboxes,
                                send_box_ranks, recv_box_ranks,
                                zzz_StructMatrixGrid(matrix),
                                zzz_StructMatrixDataSpace(matrix),
                                zzz_StructMatrixNumValues(matrix));

      zzz_StructMatrixCommPkg(matrix) = comm_pkg;

      zzz_FreeStructStencil(comm_stencil);
   }

   /*-----------------------------------------------------------------------
    * Update the ghost data
    *-----------------------------------------------------------------------*/

   comm_handle =
      zzz_InitializeCommunication(comm_pkg, zzz_StructMatrixData(matrix));
   zzz_FinalizeCommunication(comm_handle);

   return(ierr);
}

/*--------------------------------------------------------------------------
 * zzz_SetStructMatrixNumGhost
 *--------------------------------------------------------------------------*/

void
zzz_SetStructMatrixNumGhost( zzz_StructMatrix *matrix,
                             int              *num_ghost )
{
   int  i;

   for (i = 0; i < 6; i++)
      zzz_StructMatrixNumGhost(matrix)[i] = num_ghost[i];
}

/*--------------------------------------------------------------------------
 * zzz_PrintStructMatrix
 *--------------------------------------------------------------------------*/

void
zzz_PrintStructMatrix( char             *filename,
                       zzz_StructMatrix *matrix,
                       int               all      )
{
   FILE               *file;
   char                new_filename[255];

   zzz_StructGrid     *grid;
   zzz_BoxArray       *boxes;

   zzz_StructStencil  *stencil;
   zzz_Index         **stencil_shape;

   int                 num_values;

   zzz_BoxArray       *data_space;

   int                *symm_coeff;

   int                 i, j;

   int                 myid;
 
   /*----------------------------------------
    * Open file
    *----------------------------------------*/
 
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );
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

   fprintf(file, "\nSymmetric: %d\n", zzz_StructMatrixSymmetric(matrix));

   /* print grid info */
   fprintf(file, "\nGrid:\n");
   grid = zzz_StructMatrixGrid(matrix);
   zzz_PrintStructGrid(file, grid);

   /* print stencil info */
   fprintf(file, "\nStencil:\n");
   stencil = zzz_StructMatrixStencil(matrix);
   stencil_shape = zzz_StructStencilShape(stencil);

   /* symmetric case */
   num_values = zzz_StructMatrixNumValues(matrix);
   fprintf(file, "%d\n", num_values);
   if (zzz_StructMatrixSymmetric(matrix))
   {
      j = 0;
      symm_coeff = zzz_StructMatrixSymmCoeff(matrix);
      for (i = 0; i < zzz_StructStencilSize(stencil); i++)
      {
         if (!symm_coeff[i])
         {
            fprintf(file, "%d: %d %d %d\n", j++,
                    zzz_IndexX(stencil_shape[i]),
                    zzz_IndexY(stencil_shape[i]),
                    zzz_IndexZ(stencil_shape[i]));
         }
      }
   }

   /* non-symmetric case */
   else
   {
      for (i = 0; i < zzz_StructStencilSize(stencil); i++)
      {
         fprintf(file, "%d: %d %d %d\n", i,
                 zzz_IndexX(stencil_shape[i]),
                 zzz_IndexY(stencil_shape[i]),
                 zzz_IndexZ(stencil_shape[i]));
      }
   }

   /*----------------------------------------
    * Print data
    *----------------------------------------*/

   data_space = zzz_StructMatrixDataSpace(matrix);
 
   if (all)
      boxes = data_space;
   else
      boxes = zzz_StructGridBoxes(grid);
 
   fprintf(file, "\nData:\n");
   zzz_PrintBoxArrayData(file, boxes, data_space, num_values,
                         zzz_StructMatrixData(matrix));

   /*----------------------------------------
    * Close file
    *----------------------------------------*/
 
   fflush(file);
   fclose(file);
}

/*--------------------------------------------------------------------------
 * zzz_ReadStructMatrix
 *--------------------------------------------------------------------------*/

zzz_StructMatrix *
zzz_ReadStructMatrix( char *filename,
                      int  *num_ghost )
{
   FILE               *file;
   char                new_filename[255];
                      
   zzz_StructMatrix   *matrix;

   zzz_StructGrid     *grid;
   zzz_BoxArray       *boxes;
   int                 dim;

   zzz_StructStencil  *stencil;
   zzz_Index         **stencil_shape;
   int                 stencil_size;

   int                 num_values;

   zzz_BoxArray       *data_space;

   int                 symmetric;

   int                 i, idummy;

   int                 myid;
 
   /*----------------------------------------
    * Open file
    *----------------------------------------*/
 
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );
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

   /* read grid info */
   fscanf(file, "\nGrid:\n");
   grid = zzz_ReadStructGrid(file);

   /* read stencil info */
   fscanf(file, "\nStencil:\n");
   dim = zzz_StructGridDim(grid);
   fscanf(file, "%d\n", &stencil_size);
   stencil_shape = ctalloc(zzz_Index *, stencil_size);
   for (i = 0; i < stencil_size; i++)
   {
      stencil_shape[i] = zzz_NewIndex();
      fscanf(file, "%d: %d %d %d\n", &idummy,
             &zzz_IndexX(stencil_shape[i]),
             &zzz_IndexY(stencil_shape[i]),
             &zzz_IndexZ(stencil_shape[i]));
   }
   stencil = zzz_NewStructStencil(dim, stencil_size, stencil_shape);

   /*----------------------------------------
    * Initialize the matrix
    *----------------------------------------*/

   matrix = zzz_NewStructMatrix(grid, stencil);
   zzz_StructMatrixSymmetric(matrix) = symmetric;
   zzz_SetStructMatrixNumGhost(matrix, num_ghost);
   zzz_InitializeStructMatrix(matrix);

   /*----------------------------------------
    * Read data
    *----------------------------------------*/

   boxes      = zzz_StructGridBoxes(grid);
   data_space = zzz_StructMatrixDataSpace(matrix);
   num_values = zzz_StructMatrixNumValues(matrix);
 
   fscanf(file, "\nData:\n");
   zzz_ReadBoxArrayData(file, boxes, data_space, num_values,
                        zzz_StructMatrixData(matrix));

   /*----------------------------------------
    * Assemble the matrix
    *----------------------------------------*/

   zzz_AssembleStructMatrix(matrix);

   /*----------------------------------------
    * Close file
    *----------------------------------------*/
 
   fclose(file);

   return matrix;
}

