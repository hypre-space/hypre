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
 * Member functions for zzz_StructVector class.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * zzz_NewStructVector
 *--------------------------------------------------------------------------*/

zzz_StructVector *
zzz_NewStructVector( zzz_StructGrid *grid )
{
   zzz_StructVector  *vector;

   int                i;

   vector = ctalloc(zzz_StructVector, 1);

   zzz_StructVectorGrid(vector) = grid;

   /* set defaults */
   for (i = 0; i < 6; i++)
      zzz_StructVectorNumGhost(vector)[i] = 1;

   return vector;
}

/*--------------------------------------------------------------------------
 * zzz_FreeStructVector
 *--------------------------------------------------------------------------*/

int 
zzz_FreeStructVector( zzz_StructVector *vector )
{
   int  ierr;

   if (vector)
   {
      tfree(zzz_StructVectorDataIndices(vector));
      tfree(zzz_StructVectorData(vector));
 
      zzz_FreeBoxArray(zzz_StructVectorDataSpace(vector));
 
      tfree(vector);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_InitializeStructVectorShell
 *--------------------------------------------------------------------------*/

int 
zzz_InitializeStructVectorShell( zzz_StructVector *vector )
{
   int    ierr;

   zzz_StructGrid     *grid;

   int                *num_ghost;
 
   zzz_BoxArray       *data_space;
   zzz_BoxArray       *boxes;
   zzz_Box            *box;
   zzz_Box            *data_box;

   int                *data_indices;
   int                 data_size;

   int                 i, d;
 
   /*-----------------------------------------------------------------------
    * Set up data_space
    *-----------------------------------------------------------------------*/

   grid = zzz_StructVectorGrid(vector);
   num_ghost = zzz_StructVectorNumGhost(vector);

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

   zzz_StructVectorDataSpace(vector) = data_space;

   /*-----------------------------------------------------------------------
    * Set up data_indices array and data_size
    *-----------------------------------------------------------------------*/

   data_indices = ctalloc(int, zzz_BoxArraySize(data_space));

   data_size = 0;
   zzz_ForBoxI(i, data_space)
   {
      data_box = zzz_BoxArrayBox(data_space, i);

      data_indices[i] = data_size;
      data_size += zzz_BoxVolume(data_box);
   }

   zzz_StructVectorDataIndices(vector) = data_indices;
   zzz_StructVectorDataSize(vector)    = data_size;

   /*-----------------------------------------------------------------------
    * Set total number of nonzero coefficients
    *-----------------------------------------------------------------------*/

   zzz_StructVectorGlobalSize(vector) = zzz_StructGridGlobalSize(grid);

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_InitializeStructVectorData
 *--------------------------------------------------------------------------*/

void
zzz_InitializeStructVectorData( zzz_StructVector *vector,
                                double           *data   )
{
   zzz_StructVectorData(vector) = data;
}

/*--------------------------------------------------------------------------
 * zzz_InitializeStructVector
 *--------------------------------------------------------------------------*/

int 
zzz_InitializeStructVector( zzz_StructVector *vector )
{
   int    ierr;

   double *data;

   ierr = zzz_InitializeStructVectorShell(vector);

   data = ctalloc(double, zzz_StructVectorDataSize(vector));
   zzz_InitializeStructVectorData(vector, data);

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SetStructVectorValues
 *--------------------------------------------------------------------------*/

int 
zzz_SetStructVectorValues( zzz_StructVector *vector,
                           zzz_Index        *grid_index,
                           double            values     )
{
   int    ierr;

   zzz_BoxArray     *boxes;
   zzz_Box          *box;
   zzz_Index        *imin;
   zzz_Index        *imax;

   double           *vecp;

   int               i, s;

   boxes = zzz_StructGridBoxes(zzz_StructVectorGrid(vector));

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
         vecp = zzz_StructVectorBoxDataValue(vector, i, grid_index);
         *vecp = values;
      }
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SetStructVectorBoxValues
 *--------------------------------------------------------------------------*/

int 
zzz_SetStructVectorBoxValues( zzz_StructVector *vector,
                              zzz_Box          *value_box,
                              double           *values     )
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

   int               i, s, d;

   /*-----------------------------------------------------------------------
    * Set up `box_array' by intersecting `box' with the grid boxes
    *-----------------------------------------------------------------------*/

   box_array = zzz_NewBoxArray();
   grid_boxes = zzz_StructGridBoxes(zzz_StructVectorGrid(vector));
   zzz_ForBoxI(i, grid_boxes)
   {
      grid_box = zzz_BoxArrayBox(grid_boxes, i);
      box = zzz_IntersectBoxes(value_box, grid_box);
      zzz_AppendBox(box, box_array);
   }

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   if (box_array)
   {
      index = zzz_NewIndex();
 
      data_space = zzz_StructVectorDataSpace(vector);
      data_stride = zzz_NewIndex();
      zzz_IndexD(data_stride, 0) = 1;
      zzz_IndexD(data_stride, 1) = 1;
      zzz_IndexD(data_stride, 2) = 1;
 
      dval_box = zzz_DuplicateBox(value_box);
      dval_stride = zzz_NewIndex();
      zzz_IndexD(dval_stride, 0) = 1;
      zzz_IndexD(dval_stride, 1) = 1;
      zzz_IndexD(dval_stride, 2) = 1;
      dval_start = zzz_NewIndex();
 
      zzz_ForBoxI(i, box_array)
      {
         box      = zzz_BoxArrayBox(box_array, i);
         data_box = zzz_BoxArrayBox(data_space, i);
 
         /* if there was an intersection */
         if (box)
         {
            data_start = zzz_BoxIMin(box);
            for (d = 0; d < 3; d++)
               zzz_IndexD(dval_start, d) = zzz_IndexD(data_start, d);
 
            datap = zzz_StructVectorBoxData(vector, i);
 
            zzz_BoxLoop2(box, index,
                         data_box, data_start, data_stride, datai,
                         dval_box, dval_start, dval_stride, dvali,
                         {
                            datap[datai] = values[dvali];
                         });
         }
      }

      zzz_FreeBox(dval_box);
      zzz_FreeIndex(dval_start);
      zzz_FreeIndex(dval_stride);
      zzz_FreeIndex(data_stride);
      zzz_FreeIndex(index);
   }
 
   zzz_FreeBoxArray(box_array);

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_AssembleStructVector
 *--------------------------------------------------------------------------*/

int 
zzz_AssembleStructVector( zzz_StructVector *vector )
{
   int  ierr;

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SetStructVectorNumGhost
 *--------------------------------------------------------------------------*/
 
void
zzz_SetStructVectorNumGhost( zzz_StructVector *vector,
                             int              *num_ghost )
{
   int  i;
 
   for (i = 0; i < 6; i++)
      zzz_StructVectorNumGhost(vector)[i] = num_ghost[i];
}

/*--------------------------------------------------------------------------
 * zzz_PrintStructVector
 *--------------------------------------------------------------------------*/

void
zzz_PrintStructVector( char             *filename,
                       zzz_StructVector *vector,
                       int               all      )
{
   FILE            *file;
   char             new_filename[255];

   zzz_StructGrid  *grid;
   zzz_BoxArray    *boxes;

   zzz_BoxArray    *data_space;

   int              myid;
 
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

   fprintf(file, "StructVector\n");

   /* print grid info */
   fprintf(file, "\nGrid:\n");
   grid = zzz_StructVectorGrid(vector);
   zzz_PrintStructGrid(file, grid);

   /*----------------------------------------
    * Print data
    *----------------------------------------*/

   data_space = zzz_StructVectorDataSpace(vector);

   if (all)
      boxes = data_space;
   else
      boxes = zzz_StructGridBoxes(grid);

   fprintf(file, "\nData:\n");
   zzz_PrintBoxArrayData(file, boxes, data_space, 1,
                         zzz_StructVectorData(vector));
 
   /*----------------------------------------
    * Close file
    *----------------------------------------*/
 
   fflush(file);
   fclose(file);
}

/*--------------------------------------------------------------------------
 * zzz_ReadStructVector
 *--------------------------------------------------------------------------*/

zzz_StructVector *
zzz_ReadStructVector( char *filename,
                      int  *num_ghost )
{
   FILE               *file;
   char                new_filename[255];
                      
   zzz_StructVector   *vector;

   zzz_StructGrid     *grid;
   zzz_BoxArray       *boxes;

   zzz_BoxArray       *data_space;

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

   fscanf(file, "StructVector\n");

   /* read grid info */
   fscanf(file, "\nGrid:\n");
   grid = zzz_ReadStructGrid(file);

   /*----------------------------------------
    * Initialize the vector
    *----------------------------------------*/

   vector = zzz_NewStructVector(grid);
   zzz_SetStructVectorNumGhost(vector, num_ghost);
   zzz_InitializeStructVector(vector);

   /*----------------------------------------
    * Read data
    *----------------------------------------*/

   boxes      = zzz_StructGridBoxes(grid);
   data_space = zzz_StructVectorDataSpace(vector);
 
   fscanf(file, "\nData:\n");
   zzz_ReadBoxArrayData(file, boxes, data_space, 1,
                        zzz_StructVectorData(vector));

   /*----------------------------------------
    * Assemble the vector
    *----------------------------------------*/

   zzz_AssembleStructVector(vector);

   /*----------------------------------------
    * Close file
    *----------------------------------------*/
 
   fclose(file);

   return vector;
}

