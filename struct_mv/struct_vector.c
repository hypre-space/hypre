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
            zzz_BoxIMinD(data_box, d) += num_ghost[2*d + 1];
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


   zzz_BoxArray     *box_array;
   zzz_Box          *box;
   zzz_BoxArray     *box_a0;
   zzz_BoxArray     *box_a1;

   zzz_BoxArray     *data_space;
   zzz_Box          *data_box;
   zzz_Index        *index;
   zzz_Index        *stride;

   double           *vecp;
   int               veci;

   int               value_index;

   int               i, s, d;

   /*-----------------------------------------------------------------------
    * Set up `box_array' by intersecting `box' with the grid boxes
    *-----------------------------------------------------------------------*/

   box_a0 = zzz_NewBoxArray();
   zzz_AppendBox(value_box, box_a0);
   box_a1 = zzz_StructGridBoxes(zzz_StructVectorGrid(vector));

   box_array = zzz_IntersectBoxArrays(box_a0, box_a1);

   zzz_FreeBoxArrayShell(box_a0);

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   if (box_array)
   {
      data_space = zzz_StructVectorDataSpace(vector);

      index = zzz_NewIndex();

      stride = zzz_NewIndex();
      for (d = 0; d < 3; d++)
         zzz_IndexD(stride, d) = 1;

      zzz_ForBoxI(i, box_array)
      {
         box      = zzz_BoxArrayBox(box_array, i);
         data_box = zzz_BoxArrayBox(data_space, i);
 
         vecp = zzz_StructVectorBoxData(vector, i);

         value_index = 0;
         zzz_BoxLoop1(box, index,
                      data_box, zzz_BoxIMin(box), stride, veci,
                      {
                         vecp[veci] = values[value_index];
                         value_index ++;
                      });
      }

      zzz_FreeIndex(stride);
      zzz_FreeIndex(index);

      zzz_FreeBoxArray(box_array);
   }

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
 * zzz_PrintStructVector
 *--------------------------------------------------------------------------*/

void
zzz_PrintStructVector( char             *filename,
                       zzz_StructVector *vector,
                       int               all      )
{
   FILE            *file;
   char             new_filename[255];

   zzz_BoxArray        *boxes;
   zzz_BoxArray        *data_space;

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

   fprintf(file, "\nNumValues:\n");
   fprintf(file, "1\n");

   /*----------------------------------------
    * Print data
    *----------------------------------------*/

   data_space = zzz_StructVectorDataSpace(vector);

   if (all)
      boxes = data_space;
   else
      boxes = zzz_StructGridBoxes(zzz_StructVectorGrid(vector));

   fprintf(file, "\nData:\n");
   zzz_PrintBoxArrayData(file, boxes, data_space, 1,
                         zzz_StructVectorData(vector));
 
   /*----------------------------------------
    * Close file
    *----------------------------------------*/
 
   fflush(file);
   fclose(file);
}
