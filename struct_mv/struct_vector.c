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
#ifdef HYPRE_USE_PTHREADS
#include "box_pthreads.h"
#endif
/*--------------------------------------------------------------------------
 * hypre_NewStructVector
 *--------------------------------------------------------------------------*/

hypre_StructVector *
hypre_NewStructVector( MPI_Comm          comm,
                       hypre_StructGrid *grid )
{
   hypre_StructVector  *vector;

   int                  i;

   vector = hypre_CTAlloc(hypre_StructVector, 1);

   hypre_StructVectorComm(vector)        = comm;
   hypre_StructVectorGrid(vector)        = hypre_RefStructGrid(grid);
   hypre_StructVectorDataAlloced(vector) = 1;
   hypre_StructVectorRefCount(vector)    = 1;

   /* set defaults */
   for (i = 0; i < 6; i++)
      hypre_StructVectorNumGhost(vector)[i] = 1;

   return vector;
}

/*--------------------------------------------------------------------------
 * hypre_RefStructVector
 *--------------------------------------------------------------------------*/

hypre_StructVector *
hypre_RefStructVector( hypre_StructVector *vector )
{
   hypre_StructVectorRefCount(vector) ++;

   return vector;
}

/*--------------------------------------------------------------------------
 * hypre_FreeStructVector
 *--------------------------------------------------------------------------*/

int 
hypre_FreeStructVector( hypre_StructVector *vector )
{
   int  ierr = 0;

   if (vector)
   {
      hypre_StructVectorRefCount(vector) --;
      if (hypre_StructVectorRefCount(vector) == 0)
      {
         if (hypre_StructVectorDataAlloced(vector))
         {
            hypre_SharedTFree(hypre_StructVectorData(vector));
         }
         hypre_TFree(hypre_StructVectorDataIndices(vector));
         hypre_FreeBoxArray(hypre_StructVectorDataSpace(vector));
         hypre_FreeStructGrid(hypre_StructVectorGrid(vector));
         hypre_TFree(vector);
      }
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_InitializeStructVectorShell
 *--------------------------------------------------------------------------*/

int 
hypre_InitializeStructVectorShell( hypre_StructVector *vector )
{
   int    ierr = 0;

   hypre_StructGrid     *grid;

   int                  *num_ghost;
 
   hypre_BoxArray       *data_space;
   hypre_BoxArray       *boxes;
   hypre_Box            *box;
   hypre_Box            *data_box;

   int                  *data_indices;
   int                   data_size;
                     
   int                   i, d;
 
   /*-----------------------------------------------------------------------
    * Set up data_space
    *-----------------------------------------------------------------------*/

   grid = hypre_StructVectorGrid(vector);

   if (hypre_StructVectorDataSpace(vector) == NULL)
   {
      num_ghost = hypre_StructVectorNumGhost(vector);

      boxes = hypre_StructGridBoxes(grid);
      data_space = hypre_NewBoxArray(hypre_BoxArraySize(boxes));

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

      hypre_StructVectorDataSpace(vector) = data_space;
   }

   /*-----------------------------------------------------------------------
    * Set up data_indices array and data_size
    *-----------------------------------------------------------------------*/

   if (hypre_StructVectorDataIndices(vector) == NULL)
   {
      data_space = hypre_StructVectorDataSpace(vector);
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
   }

   /*-----------------------------------------------------------------------
    * Set total number of nonzero coefficients
    *-----------------------------------------------------------------------*/

   hypre_StructVectorGlobalSize(vector) = hypre_StructGridGlobalSize(grid);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_InitializeStructVectorData
 *--------------------------------------------------------------------------*/

int
hypre_InitializeStructVectorData( hypre_StructVector *vector,
                                  double             *data   )
{
   int ierr = 0;

   hypre_StructVectorData(vector) = data;
   hypre_StructVectorDataAlloced(vector) = 0;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_InitializeStructVector
 *--------------------------------------------------------------------------*/

int 
hypre_InitializeStructVector( hypre_StructVector *vector )
{
   int    ierr = 0;

   double *data;

   ierr = hypre_InitializeStructVectorShell(vector);

   data = hypre_SharedCTAlloc(double, hypre_StructVectorDataSize(vector));
   hypre_InitializeStructVectorData(vector, data);
   hypre_StructVectorDataAlloced(vector) = 1;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SetStructVectorValues
 *--------------------------------------------------------------------------*/

int 
hypre_SetStructVectorValues( hypre_StructVector *vector,
                             hypre_Index         grid_index,
                             double              values     )
{
   int    ierr = 0;

   hypre_BoxArray     *boxes;
   hypre_Box          *box;

   double             *vecp;

   int                 i;

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
                             double             *values_ptr )
{
   int    ierr = 0;

   double              values;

   hypre_BoxArray     *boxes;
   hypre_Box          *box;

   double             *vecp;

   int                 i;

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
            values = *vecp;
         }
      }

   *values_ptr = values;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SetStructVectorBoxValues
 *--------------------------------------------------------------------------*/

int 
hypre_SetStructVectorBoxValues( hypre_StructVector *vector,
                                hypre_Box          *value_box,
                                double             *values     )
{
   int    ierr = 0;

   hypre_BoxArray     *grid_boxes;
   hypre_Box          *grid_box;
   hypre_BoxArray     *box_array;
   hypre_Box          *box;

   hypre_BoxArray     *data_space;
   hypre_Box          *data_box;
   hypre_IndexRef      data_start;
   hypre_Index         data_stride;
   int                 datai;
   double             *datap;

   hypre_Box          *dval_box;
   hypre_Index         dval_start;
   hypre_Index         dval_stride;
   int                 dvali;

   hypre_Index         loop_size;

   int                 i;
   int                 loopi, loopj, loopk;

   /*-----------------------------------------------------------------------
    * Set up `box_array' by intersecting `box' with the grid boxes
    *-----------------------------------------------------------------------*/

   grid_boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(vector));
   box_array = hypre_NewBoxArray(hypre_BoxArraySize(grid_boxes));
   box = hypre_NewBox();
   hypre_ForBoxI(i, grid_boxes)
      {
         grid_box = hypre_BoxArrayBox(grid_boxes, i);
         hypre_IntersectBoxes(value_box, grid_box, box);
         hypre_CopyBox(box, hypre_BoxArrayBox(box_array, i));
      }
   hypre_FreeBox(box);

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
 * hypre_GetStructVectorBoxValues
 *--------------------------------------------------------------------------*/

int 
hypre_GetStructVectorBoxValues( hypre_StructVector *vector,
                                hypre_Box          *value_box,
                                double             *values    )
{
   int    ierr = 0;

   hypre_BoxArray     *grid_boxes;
   hypre_Box          *grid_box;
   hypre_BoxArray     *box_array;
   hypre_Box          *box;

   hypre_BoxArray     *data_space;
   hypre_Box          *data_box;
   hypre_IndexRef      data_start;
   hypre_Index         data_stride;
   int                 datai;
   double             *datap;

   hypre_Box          *dval_box;
   hypre_Index         dval_start;
   hypre_Index         dval_stride;
   int                 dvali;

   hypre_Index         loop_size;

   int                 i;
   int                 loopi, loopj, loopk;

   /*-----------------------------------------------------------------------
    * Set up `box_array' by intersecting `box' with the grid boxes
    *-----------------------------------------------------------------------*/

   grid_boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(vector));
   box_array = hypre_NewBoxArray(hypre_BoxArraySize(grid_boxes));
   box = hypre_NewBox();
   hypre_ForBoxI(i, grid_boxes)
      {
         grid_box = hypre_BoxArrayBox(grid_boxes, i);
         hypre_IntersectBoxes(value_box, grid_box, box);
         hypre_CopyBox(box, hypre_BoxArrayBox(box_array, i));
      }
   hypre_FreeBox(box);

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
                                 values[dvali] = datap[datai];
                              });
            }
         }

      hypre_FreeBox(dval_box);
   }
 
   hypre_FreeBoxArray(box_array);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SetStructVectorNumGhost
 *--------------------------------------------------------------------------*/
 
int
hypre_SetStructVectorNumGhost( hypre_StructVector *vector,
                               int                *num_ghost )
{
   int  ierr = 0;
   int  i;
 
   for (i = 0; i < 6; i++)
      hypre_StructVectorNumGhost(vector)[i] = num_ghost[i];

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_AssembleStructVector
 *--------------------------------------------------------------------------*/

int 
hypre_AssembleStructVector( hypre_StructVector *vector )
{
   int  ierr = 0;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SetStructVectorConstantValues
 *--------------------------------------------------------------------------*/

int 
hypre_SetStructVectorConstantValues( hypre_StructVector *vector,
                                     double              values )
{
   int    ierr = 0;

   hypre_Box          *v_data_box;
                    
   int                 vi;
   double             *vp;

   hypre_BoxArray     *boxes;
   hypre_Box          *box;
   hypre_Index         loop_size;
   hypre_IndexRef      start;
   hypre_Index         unit_stride;

   int                 i;
   int                 loopi, loopj, loopk;

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   hypre_SetIndex(unit_stride, 1, 1, 1);
 
   boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(vector));
   hypre_ForBoxI(i, boxes)
      {
         box      = hypre_BoxArrayBox(boxes, i);
         start = hypre_BoxIMin(box);

         v_data_box =
            hypre_BoxArrayBox(hypre_StructVectorDataSpace(vector), i);
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
   int    ierr = 0;

   hypre_Box          *v_data_box;
                    
   int                 vi;
   double             *vp;

   hypre_BoxArray     *boxes;
   hypre_Box          *box;
   hypre_BoxArray     *diff_boxes;
   hypre_Box          *diff_box;
   hypre_Index         loop_size;
   hypre_IndexRef      start;
   hypre_Index         unit_stride;

   int                 i, j;
   int                 loopi, loopj, loopk;

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   hypre_SetIndex(unit_stride, 1, 1, 1);
 
   boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(vector));
   diff_boxes = hypre_NewBoxArray(0);
   hypre_ForBoxI(i, boxes)
      {
         box        = hypre_BoxArrayBox(boxes, i);

         v_data_box =
            hypre_BoxArrayBox(hypre_StructVectorDataSpace(vector), i);
         vp = hypre_StructVectorBoxData(vector, i);

         hypre_SubtractBoxes(v_data_box, box, diff_boxes);
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
      }
   hypre_FreeBoxArray(diff_boxes);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ClearStructVectorAllValues
 *--------------------------------------------------------------------------*/

int 
hypre_ClearStructVectorAllValues( hypre_StructVector *vector )
{
   int               ierr = 0;

   int               data_size;
   double           *data;

   int               i;
   int               loopi, loopj, loopk;
   hypre_Index       loop_size;

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   data_size = hypre_StructVectorDataSize(vector);
   data      = hypre_StructVectorData(vector);

   hypre_SetIndex(loop_size, data_size, 1, 1);

   hypre_BoxLoop0(loopi, loopk, loopj, loop_size,
                  {
                     data[loopi] = 0.0;
                  });

#if 0
   for ( i=0; i < data_size; i++)
      data[i] = 0.0;

#ifdef HYPRE_USE_PTHREADS
   hypre_barrier(&hypre_mutex_boxloops, 0);
#endif

#endif

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_GetMigrateStructVectorCommPkg
 *--------------------------------------------------------------------------*/

hypre_CommPkg *
hypre_GetMigrateStructVectorCommPkg( hypre_StructVector *from_vector,
				     hypre_StructVector *to_vector   )
{
   hypre_BoxArrayArray   *send_boxes;
   hypre_BoxArrayArray   *recv_boxes;
   int                  **send_processes;
   int                  **recv_processes;
   int                    num_values;

   hypre_Index            unit_stride;

   hypre_CommPkg         *comm_pkg;

   /*------------------------------------------------------
    * Set up hypre_CommPkg
    *------------------------------------------------------*/
 
   num_values = 1;
   hypre_SetIndex(unit_stride, 1, 1, 1);

   hypre_NewCommInfoFromGrids(hypre_StructVectorGrid(from_vector),
                              hypre_StructVectorGrid(to_vector),
                              &send_boxes, &recv_boxes,
                              &send_processes, &recv_processes);

   comm_pkg = hypre_NewCommPkg(send_boxes, recv_boxes,
                               unit_stride, unit_stride,
                               hypre_StructVectorDataSpace(from_vector),
                               hypre_StructVectorDataSpace(to_vector),
                               send_processes, recv_processes,
                               num_values,
                               hypre_StructVectorComm(from_vector));

   return comm_pkg;
}

/*--------------------------------------------------------------------------
 * hypre_MigrateStructVector
 *--------------------------------------------------------------------------*/

int 
hypre_MigrateStructVector( hypre_CommPkg      *comm_pkg,
			   hypre_StructVector *from_vector,
                           hypre_StructVector *to_vector   )
{
   hypre_CommHandle      *comm_handle;

   int                    ierr = 0;

   /*-----------------------------------------------------------------------
    * Migrate the vector data
    *-----------------------------------------------------------------------*/
 
   hypre_InitializeCommunication(comm_pkg,
                                 hypre_StructVectorData(from_vector),
                                 hypre_StructVectorData(to_vector),
                                 &comm_handle);
   hypre_FinalizeCommunication(comm_handle);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PrintStructVector
 *--------------------------------------------------------------------------*/

int
hypre_PrintStructVector( char               *filename,
                         hypre_StructVector *vector,
                         int                 all      )
{
   int                ierr = 0;

   FILE              *file;
   char               new_filename[255];

   hypre_StructGrid  *grid;
   hypre_BoxArray    *boxes;

   hypre_BoxArray    *data_space;

   int                myid;

   /*----------------------------------------
    * Open file
    *----------------------------------------*/

   MPI_Comm_rank(hypre_StructVectorComm(vector), &myid );

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

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ReadStructVector
 *--------------------------------------------------------------------------*/

hypre_StructVector *
hypre_ReadStructVector( MPI_Comm   comm,
                        char      *filename,
                        int       *num_ghost )
{
   FILE                 *file;
   char                  new_filename[255];
                      
   hypre_StructVector   *vector;

   hypre_StructGrid     *grid;
   hypre_BoxArray       *boxes;

   hypre_BoxArray       *data_space;

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

