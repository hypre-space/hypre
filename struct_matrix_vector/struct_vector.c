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
 * Member functions for hypre_StructVector class.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_NewStructVector
 *--------------------------------------------------------------------------*/

hypre_StructVector *
hypre_NewStructVector( MPI_Comm       *comm,
                     hypre_StructGrid *grid )
{
   hypre_StructVector  *vector;

   int                i;

   vector = hypre_CTAlloc(hypre_StructVector, 1);

   hypre_StructVectorComm(vector) = comm;
   hypre_StructVectorGrid(vector) = grid;

   /* set defaults */
   for (i = 0; i < 6; i++)
      hypre_StructVectorNumGhost(vector)[i] = 1;

   return vector;
}

/*--------------------------------------------------------------------------
 * hypre_FreeStructVectorShell
 *--------------------------------------------------------------------------*/

int 
hypre_FreeStructVectorShell( hypre_StructVector *vector )
{
   int  ierr;

   if (vector)
   {
      hypre_TFree(hypre_StructVectorDataIndices(vector));
      hypre_FreeBoxArray(hypre_StructVectorDataSpace(vector));
      hypre_TFree(vector);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_FreeStructVector
 *--------------------------------------------------------------------------*/

int 
hypre_FreeStructVector( hypre_StructVector *vector )
{
   int  ierr;

   if (vector)
   {
      hypre_TFree(hypre_StructVectorData(vector));
      hypre_FreeStructVectorShell(vector);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_InitializeStructVectorShell
 *--------------------------------------------------------------------------*/

int 
hypre_InitializeStructVectorShell( hypre_StructVector *vector )
{
   int    ierr;

   hypre_StructGrid     *grid;

   int                *num_ghost;
 
   hypre_BoxArray       *data_space;
   hypre_BoxArray       *boxes;
   hypre_Box            *box;
   hypre_Box            *data_box;

   int                *data_indices;
   int                 data_size;

   int                 i, d;
 
   /*-----------------------------------------------------------------------
    * Set up data_space
    *-----------------------------------------------------------------------*/

   grid = hypre_StructVectorGrid(vector);
   num_ghost = hypre_StructVectorNumGhost(vector);

   boxes = hypre_StructGridBoxes(grid);
   data_space = hypre_NewBoxArray();

   hypre_ForBoxI(i, boxes)
   {
      box = hypre_BoxArrayBox(boxes, i);

      data_box = hypre_DuplicateBox(box);
      if (hypre_BoxVolume(data_box))
      {
         for (d = 0; d < 3; d++)
         {
            hypre_BoxIMinD(data_box, d) -= num_ghost[2*d];
            hypre_BoxIMaxD(data_box, d) += num_ghost[2*d + 1];
         }
      }

      hypre_AppendBox(data_box, data_space);
   }

   hypre_StructVectorDataSpace(vector) = data_space;

   /*-----------------------------------------------------------------------
    * Set up data_indices array and data_size
    *-----------------------------------------------------------------------*/

   data_indices = hypre_CTAlloc(int, hypre_BoxArraySize(data_space));

   data_size = 0;
   hypre_ForBoxI(i, data_space)
   {
      data_box = hypre_BoxArrayBox(data_space, i);

      data_indices[i] = data_size;
      data_size += hypre_BoxVolume(data_box);
   }

   hypre_StructVectorDataIndices(vector) = data_indices;
   hypre_StructVectorDataSize(vector)    = data_size;

   /*-----------------------------------------------------------------------
    * Set total number of nonzero coefficients
    *-----------------------------------------------------------------------*/

   hypre_StructVectorGlobalSize(vector) = hypre_StructGridGlobalSize(grid);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_InitializeStructVectorData
 *--------------------------------------------------------------------------*/

void
hypre_InitializeStructVectorData( hypre_StructVector *vector,
                                double           *data   )
{
   hypre_StructVectorData(vector) = data;
}

/*--------------------------------------------------------------------------
 * hypre_InitializeStructVector
 *--------------------------------------------------------------------------*/

int 
hypre_InitializeStructVector( hypre_StructVector *vector )
{
   int    ierr;

   double *data;

   ierr = hypre_InitializeStructVectorShell(vector);

   data = hypre_CTAlloc(double, hypre_StructVectorDataSize(vector));
   hypre_InitializeStructVectorData(vector, data);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SetStructVectorValues
 *--------------------------------------------------------------------------*/

int 
hypre_SetStructVectorValues( hypre_StructVector *vector,
                           hypre_Index         grid_index,
                           double            values     )
{
   int    ierr;

   hypre_BoxArray     *boxes;
   hypre_Box          *box;

   double           *vecp;

   int               i;

   boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(vector));

   hypre_ForBoxI(i, boxes)
   {
      box = hypre_BoxArrayBox(boxes, i);

      if ((hypre_IndexX(grid_index) >= hypre_BoxIMinX(box)) &&
          (hypre_IndexX(grid_index) <= hypre_BoxIMaxX(box)) &&
          (hypre_IndexY(grid_index) >= hypre_BoxIMinY(box)) &&
          (hypre_IndexY(grid_index) <= hypre_BoxIMaxY(box)) &&
          (hypre_IndexZ(grid_index) >= hypre_BoxIMinZ(box)) &&
          (hypre_IndexZ(grid_index) <= hypre_BoxIMaxZ(box))   )
      {
         vecp = hypre_StructVectorBoxDataValue(vector, i, grid_index);
         *vecp = values;
      }
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_GetStructVectorValues
 *--------------------------------------------------------------------------*/

int 
hypre_GetStructVectorValues( hypre_StructVector *vector,
                           hypre_Index         grid_index,
                           double           *values     )
{
   int    ierr;

   hypre_BoxArray     *boxes;
   hypre_Box          *box;

   double           *vecp;

   int               i;

   boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(vector));

   hypre_ForBoxI(i, boxes)
   {
      box = hypre_BoxArrayBox(boxes, i);

      if ((hypre_IndexX(grid_index) >= hypre_BoxIMinX(box)) &&
          (hypre_IndexX(grid_index) <= hypre_BoxIMaxX(box)) &&
          (hypre_IndexY(grid_index) >= hypre_BoxIMinY(box)) &&
          (hypre_IndexY(grid_index) <= hypre_BoxIMaxY(box)) &&
          (hypre_IndexZ(grid_index) >= hypre_BoxIMinZ(box)) &&
          (hypre_IndexZ(grid_index) <= hypre_BoxIMaxZ(box))   )
      {
         vecp = hypre_StructVectorBoxDataValue(vector, i, grid_index);
         *values = *vecp;
      }
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SetStructVectorBoxValues
 *--------------------------------------------------------------------------*/

int 
hypre_SetStructVectorBoxValues( hypre_StructVector *vector,
                              hypre_Box          *value_box,
                              double           *values     )
{
   int    ierr;

   hypre_BoxArray     *grid_boxes;
   hypre_Box          *grid_box;
   hypre_BoxArray     *box_array;
   hypre_Box          *box;

   hypre_BoxArray     *data_space;
   hypre_Box          *data_box;
   hypre_IndexRef      data_start;
   hypre_Index         data_stride;
   int               datai;
   double           *datap;

   hypre_Box          *dval_box;
   hypre_Index         dval_start;
   hypre_Index         dval_stride;
   int               dvali;

   hypre_Index         loop_size;

   int               i;
   int               loopi, loopj, loopk;

   /*-----------------------------------------------------------------------
    * Set up `box_array' by intersecting `box' with the grid boxes
    *-----------------------------------------------------------------------*/

   box_array = hypre_NewBoxArray();
   grid_boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(vector));
   hypre_ForBoxI(i, grid_boxes)
   {
      grid_box = hypre_BoxArrayBox(grid_boxes, i);
      box = hypre_IntersectBoxes(value_box, grid_box);
      hypre_AppendBox(box, box_array);
   }

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   if (box_array)
   {
      data_space = hypre_StructVectorDataSpace(vector);
      hypre_SetIndex(data_stride, 1, 1, 1);
 
      dval_box = hypre_DuplicateBox(value_box);
      hypre_SetIndex(dval_stride, 1, 1, 1);
 
      hypre_ForBoxI(i, box_array)
      {
         box      = hypre_BoxArrayBox(box_array, i);
         data_box = hypre_BoxArrayBox(data_space, i);
 
         /* if there was an intersection */
         if (box)
         {
            data_start = hypre_BoxIMin(box);
            hypre_CopyIndex(data_start, dval_start);
 
            datap = hypre_StructVectorBoxData(vector, i);
 
            hypre_GetBoxSize(box, loop_size);
            hypre_BoxLoop2(loopi, loopj, loopk, loop_size,
                         data_box, data_start, data_stride, datai,
                         dval_box, dval_start, dval_stride, dvali,
                         {
                            datap[datai] = values[dvali];
                         });
         }
      }

      hypre_FreeBox(dval_box);
   }
 
   hypre_FreeBoxArray(box_array);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SetStructVectorConstantValues
 *--------------------------------------------------------------------------*/

int 
hypre_SetStructVectorConstantValues( hypre_StructVector *vector,
                                   double            values )
{
   int    ierr;

   hypre_Box          *v_data_box;
                    
   int               vi;
   double           *vp;

   hypre_BoxArray     *boxes;
   hypre_Box          *box;
   hypre_Index         loop_size;
   hypre_IndexRef      start;
   hypre_Index         unit_stride;

   int               i;
   int               loopi, loopj, loopk;

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   hypre_SetIndex(unit_stride, 1, 1, 1);
 
   boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(vector));
   hypre_ForBoxI(i, boxes)
   {
      box      = hypre_BoxArrayBox(boxes, i);
      start = hypre_BoxIMin(box);

      v_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(vector), i);
      vp = hypre_StructVectorBoxData(vector, i);
 
      hypre_GetBoxSize(box, loop_size);
      hypre_BoxLoop1(loopi, loopj, loopk, loop_size,
                   v_data_box, start, unit_stride, vi,
                   {
                      vp[vi] = values;
                   });
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ClearStructVectorGhostValues
 *--------------------------------------------------------------------------*/

int 
hypre_ClearStructVectorGhostValues( hypre_StructVector *vector )
{
   int    ierr;

   hypre_Box          *v_data_box;
                    
   int               vi;
   double           *vp;

   hypre_BoxArray     *boxes;
   hypre_Box          *box;
   hypre_BoxArray     *diff_boxes;
   hypre_Box          *diff_box;
   hypre_Index         loop_size;
   hypre_IndexRef      start;
   hypre_Index         unit_stride;

   int               i, j;
   int               loopi, loopj, loopk;

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   hypre_SetIndex(unit_stride, 1, 1, 1);
 
   boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(vector));
   hypre_ForBoxI(i, boxes)
   {
      box        = hypre_BoxArrayBox(boxes, i);

      v_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(vector), i);
      vp = hypre_StructVectorBoxData(vector, i);

      diff_boxes = hypre_SubtractBoxes(v_data_box, box);
      hypre_ForBoxI(j, diff_boxes)
      {
         diff_box = hypre_BoxArrayBox(diff_boxes, j);
         start = hypre_BoxIMin(diff_box);

         hypre_GetBoxSize(diff_box, loop_size);
         hypre_BoxLoop1(loopi, loopj, loopk, loop_size,
                      v_data_box, start, unit_stride, vi,
                      {
                         vp[vi] = 0.0;
                      });
      }
      hypre_FreeBoxArray(diff_boxes);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ClearStructVectorAllValues
 *--------------------------------------------------------------------------*/

int 
hypre_ClearStructVectorAllValues( hypre_StructVector *vector )
{
   int               ierr;

   int               data_size;
   double           *data;

   int               i;

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   data_size = hypre_StructVectorDataSize(vector);
   data      = hypre_StructVectorData(vector);

   for ( i=0; i < data_size; i++)
      data[i] = 0.0;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_AssembleStructVector
 *--------------------------------------------------------------------------*/

int 
hypre_AssembleStructVector( hypre_StructVector *vector )
{
   int  ierr;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SetStructVectorNumGhost
 *--------------------------------------------------------------------------*/
 
void
hypre_SetStructVectorNumGhost( hypre_StructVector *vector,
                             int              *num_ghost )
{
   int  i;
 
   for (i = 0; i < 6; i++)
      hypre_StructVectorNumGhost(vector)[i] = num_ghost[i];
}

/*--------------------------------------------------------------------------
 * hypre_PrintStructVector
 *--------------------------------------------------------------------------*/

void
hypre_PrintStructVector( char             *filename,
                       hypre_StructVector *vector,
                       int               all      )
{
   FILE            *file;
   char             new_filename[255];

   hypre_StructGrid  *grid;
   hypre_BoxArray    *boxes;

   hypre_BoxArray    *data_space;

   int              myid;

   /*----------------------------------------
    * Open file
    *----------------------------------------*/
 
   MPI_Comm_rank(*hypre_StructVectorComm(vector), &myid );
   sprintf(new_filename, "%s.%05d", filename, myid);
 
   if ((file = fopen(new_filename, "w")) == NULL)
   {
      printf("Error: can't open output file %s\n", new_filename);
      exit(1);
   }

   /*----------------------------------------
    * Print header info
    *----------------------------------------*/

   fprintf(file, "StructVector\n");

   /* print grid info */
   fprintf(file, "\nGrid:\n");
   grid = hypre_StructVectorGrid(vector);
   hypre_PrintStructGrid(file, grid);

   /*----------------------------------------
    * Print data
    *----------------------------------------*/

   data_space = hypre_StructVectorDataSpace(vector);

   if (all)
      boxes = data_space;
   else
      boxes = hypre_StructGridBoxes(grid);

   fprintf(file, "\nData:\n");
   hypre_PrintBoxArrayData(file, boxes, data_space, 1,
                         hypre_StructVectorData(vector));
 
   /*----------------------------------------
    * Close file
    *----------------------------------------*/
 
   fflush(file);
   fclose(file);
}

/*--------------------------------------------------------------------------
 * hypre_ReadStructVector
 *--------------------------------------------------------------------------*/

hypre_StructVector *
hypre_ReadStructVector( MPI_Comm *comm,
		      char *filename,
                      int  *num_ghost )
{
   FILE               *file;
   char                new_filename[255];
                      
   hypre_StructVector   *vector;

   hypre_StructGrid     *grid;
   hypre_BoxArray       *boxes;

   hypre_BoxArray       *data_space;

   int                 myid;
 
   /*----------------------------------------
    * Open file
    *----------------------------------------*/
 
   MPI_Comm_rank(*comm, &myid );
   sprintf(new_filename, "%s.%05d", filename, myid);
 
   if ((file = fopen(new_filename, "r")) == NULL)
   {
      printf("Error: can't open output file %s\n", new_filename);
      exit(1);
   }

   /*----------------------------------------
    * Read header info
    *----------------------------------------*/

   fscanf(file, "StructVector\n");

   /* read grid info */
   fscanf(file, "\nGrid:\n");
   grid = hypre_ReadStructGrid(comm,file);

   /*----------------------------------------
    * Initialize the vector
    *----------------------------------------*/

   vector = hypre_NewStructVector(comm, grid);
   hypre_SetStructVectorNumGhost(vector, num_ghost);
   hypre_InitializeStructVector(vector);

   /*----------------------------------------
    * Read data
    *----------------------------------------*/

   boxes      = hypre_StructGridBoxes(grid);
   data_space = hypre_StructVectorDataSpace(vector);
 
   fscanf(file, "\nData:\n");
   hypre_ReadBoxArrayData(file, boxes, data_space, 1,
                        hypre_StructVectorData(vector));

   /*----------------------------------------
    * Assemble the vector
    *----------------------------------------*/

   hypre_AssembleStructVector(vector);

   /*----------------------------------------
    * Close file
    *----------------------------------------*/
 
   fclose(file);

   return vector;
}

