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
 * hypre_StructVectorCreate
 *--------------------------------------------------------------------------*/

hypre_StructVector *
hypre_StructVectorCreate( MPI_Comm          comm,
                          hypre_StructGrid *grid )
{
   hypre_StructVector  *vector;

   int                  i;

   vector = hypre_CTAlloc(hypre_StructVector, 1);

   hypre_StructVectorComm(vector)        = comm;
   hypre_StructGridRef(grid, &hypre_StructVectorGrid(vector));
   hypre_StructVectorDataAlloced(vector) = 1;
   hypre_StructVectorRefCount(vector)    = 1;

   /* set defaults */
   for (i = 0; i < 6; i++)
      hypre_StructVectorNumGhost(vector)[i] = 1;

   return vector;
}

/*--------------------------------------------------------------------------
 * hypre_StructVectorRef
 *--------------------------------------------------------------------------*/

hypre_StructVector *
hypre_StructVectorRef( hypre_StructVector *vector )
{
   hypre_StructVectorRefCount(vector) ++;

   return vector;
}

/*--------------------------------------------------------------------------
 * hypre_StructVectorDestroy
 *--------------------------------------------------------------------------*/

int 
hypre_StructVectorDestroy( hypre_StructVector *vector )
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
         hypre_BoxArrayDestroy(hypre_StructVectorDataSpace(vector));
         hypre_StructGridDestroy(hypre_StructVectorGrid(vector));
         hypre_TFree(vector);
      }
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructVectorInitializeShell
 *--------------------------------------------------------------------------*/

int 
hypre_StructVectorInitializeShell( hypre_StructVector *vector )
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
 * hypre_StructVectorInitializeData
 *--------------------------------------------------------------------------*/

int
hypre_StructVectorInitializeData( hypre_StructVector *vector,
                                  double             *data   )
{
   int ierr = 0;

   hypre_StructVectorData(vector) = data;
   hypre_StructVectorDataAlloced(vector) = 0;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructVectorInitialize
 *--------------------------------------------------------------------------*/

int 
hypre_StructVectorInitialize( hypre_StructVector *vector )
{
   int    ierr = 0;

   double *data;

   ierr = hypre_StructVectorInitializeShell(vector);

   data = hypre_SharedCTAlloc(double, hypre_StructVectorDataSize(vector));
   hypre_StructVectorInitializeData(vector, data);
   hypre_StructVectorDataAlloced(vector) = 1;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructVectorSetValues
 *--------------------------------------------------------------------------*/

int 
hypre_StructVectorSetValues( hypre_StructVector *vector,
                             hypre_Index         grid_index,
                             double              values,
                             int                 add_to     )
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
            if (add_to)
            {
               *vecp += values;
            }
            else
            {
               *vecp = values;
            }
         }
      }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructVectorSetBoxValues
 *--------------------------------------------------------------------------*/

int 
hypre_StructVectorSetBoxValues( hypre_StructVector *vector,
                                hypre_Box          *value_box,
                                double             *values,
                                int                 add_to    )
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
   box_array = hypre_BoxArrayCreate(hypre_BoxArraySize(grid_boxes));
   box = hypre_BoxCreate();
   hypre_ForBoxI(i, grid_boxes)
      {
         grid_box = hypre_BoxArrayBox(grid_boxes, i);
         hypre_IntersectBoxes(value_box, grid_box, box);
         hypre_CopyBox(box, hypre_BoxArrayBox(box_array, i));
      }
   hypre_BoxDestroy(box);

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   if (box_array)
   {
      data_space = hypre_StructVectorDataSpace(vector);
      hypre_SetIndex(data_stride, 1, 1, 1);
 
      dval_box = hypre_BoxDuplicate(value_box);
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
 
               hypre_BoxGetSize(box, loop_size);

               if (add_to)
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
               else
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
            }
         }

      hypre_BoxDestroy(dval_box);
   }
 
   hypre_BoxArrayDestroy(box_array);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructVectorGetValues
 *--------------------------------------------------------------------------*/

int 
hypre_StructVectorGetValues( hypre_StructVector *vector,
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
 * hypre_StructVectorGetBoxValues
 *--------------------------------------------------------------------------*/

int 
hypre_StructVectorGetBoxValues( hypre_StructVector *vector,
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
   box_array = hypre_BoxArrayCreate(hypre_BoxArraySize(grid_boxes));
   box = hypre_BoxCreate();
   hypre_ForBoxI(i, grid_boxes)
      {
         grid_box = hypre_BoxArrayBox(grid_boxes, i);
         hypre_IntersectBoxes(value_box, grid_box, box);
         hypre_CopyBox(box, hypre_BoxArrayBox(box_array, i));
      }
   hypre_BoxDestroy(box);

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   if (box_array)
   {
      data_space = hypre_StructVectorDataSpace(vector);
      hypre_SetIndex(data_stride, 1, 1, 1);
 
      dval_box = hypre_BoxDuplicate(value_box);
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
 
               hypre_BoxGetSize(box, loop_size);

               hypre_BoxLoop2Begin(loop_size,
                                   data_box, data_start, data_stride, datai,
                                   dval_box, dval_start, dval_stride, dvali);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,datai,dvali
#include "hypre_box_smp_forloop.h"
               hypre_BoxLoop2For(loopi, loopj, loopk, datai, dvali)
                  {
                     values[dvali] = datap[datai];
                  }
               hypre_BoxLoop2End(datai, dvali);
            }
         }

      hypre_BoxDestroy(dval_box);
   }
 
   hypre_BoxArrayDestroy(box_array);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructVectorSetNumGhost
 *--------------------------------------------------------------------------*/
 
int
hypre_StructVectorSetNumGhost( hypre_StructVector *vector,
                               int                *num_ghost )
{
   int  ierr = 0;
   int  i;
 
   for (i = 0; i < 6; i++)
      hypre_StructVectorNumGhost(vector)[i] = num_ghost[i];

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructVectorAssemble
 *--------------------------------------------------------------------------*/

int 
hypre_StructVectorAssemble( hypre_StructVector *vector )
{
   int  ierr = 0;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructVectorSetConstantValues
 *--------------------------------------------------------------------------*/

int 
hypre_StructVectorSetConstantValues( hypre_StructVector *vector,
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
 
         hypre_BoxGetSize(box, loop_size);

         hypre_BoxLoop1Begin(loop_size,
                             v_data_box, start, unit_stride, vi);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,vi 
#include "hypre_box_smp_forloop.h"
         hypre_BoxLoop1For(loopi, loopj, loopk, vi)
            {
               vp[vi] = values;
            }
         hypre_BoxLoop1End(vi);
      }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructVectorClearGhostValues
 *--------------------------------------------------------------------------*/

int 
hypre_StructVectorClearGhostValues( hypre_StructVector *vector )
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
   diff_boxes = hypre_BoxArrayCreate(0);
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

               hypre_BoxGetSize(diff_box, loop_size);

               hypre_BoxLoop1Begin(loop_size,
                                   v_data_box, start, unit_stride, vi);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,vi 
#include "hypre_box_smp_forloop.h"
               hypre_BoxLoop1For(loopi, loopj, loopk, vi)
                  {
                     vp[vi] = 0.0;
                  }
               hypre_BoxLoop1End(vi);
            }
      }
   hypre_BoxArrayDestroy(diff_boxes);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructVectorClearAllValues
 *--------------------------------------------------------------------------*/

int 
hypre_StructVectorClearAllValues( hypre_StructVector *vector )
{
   int               ierr = 0;

   int               datai;
   double           *data;

   hypre_Index       imin;
   hypre_Index       imax;
   hypre_Box        *box;
   hypre_Index       loop_size;

   int               loopi, loopj, loopk;

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   box = hypre_BoxCreate();
   hypre_SetIndex(imin, 1, 1, 1);
   hypre_SetIndex(imax, hypre_StructVectorDataSize(vector), 1, 1);
   hypre_BoxSetExtents(box, imin, imax);
   data = hypre_StructVectorData(vector);
   hypre_BoxGetSize(box, loop_size);

   hypre_BoxLoop1Begin(loop_size,
                       box, imin, imin, datai);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,datai
#include "hypre_box_smp_forloop.h"
   hypre_BoxLoop1For(loopi, loopj, loopk, datai)
      {
         data[datai] = 0.0;
      }
   hypre_BoxLoop1End(datai);

   hypre_BoxDestroy(box);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructVectorGetMigrateCommPkg
 *--------------------------------------------------------------------------*/

hypre_CommPkg *
hypre_StructVectorGetMigrateCommPkg( hypre_StructVector *from_vector,
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

   hypre_CreateCommInfoFromGrids(hypre_StructVectorGrid(from_vector),
                                 hypre_StructVectorGrid(to_vector),
                                 &send_boxes, &recv_boxes,
                                 &send_processes, &recv_processes);

   comm_pkg = hypre_CommPkgCreate(send_boxes, recv_boxes,
                                  unit_stride, unit_stride,
                                  hypre_StructVectorDataSpace(from_vector),
                                  hypre_StructVectorDataSpace(to_vector),
                                  send_processes, recv_processes,
                                  num_values,
                                  hypre_StructVectorComm(from_vector),
                                  hypre_StructGridPeriodic(
                                     hypre_StructVectorGrid(from_vector)));
   /* is this correct for periodic? */

   return comm_pkg;
}

/*--------------------------------------------------------------------------
 * hypre_StructVectorMigrate
 *--------------------------------------------------------------------------*/

int 
hypre_StructVectorMigrate( hypre_CommPkg      *comm_pkg,
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
 * hypre_StructVectorPrint
 *--------------------------------------------------------------------------*/

int
hypre_StructVectorPrint( char               *filename,
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
   hypre_StructGridPrint(file, grid);

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
 * hypre_StructVectorRead
 *--------------------------------------------------------------------------*/

hypre_StructVector *
hypre_StructVectorRead( MPI_Comm   comm,
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
   hypre_StructGridRead(comm,file,&grid);

   /*----------------------------------------
    * Initialize the vector
    *----------------------------------------*/

   vector = hypre_StructVectorCreate(comm, grid);
   hypre_StructVectorSetNumGhost(vector, num_ghost);
   hypre_StructVectorInitialize(vector);

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

   hypre_StructVectorAssemble(vector);

   /*----------------------------------------
    * Close file
    *----------------------------------------*/
 
   fclose(file);

   return vector;
}

