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
   for (i = 0; i < 6; i++)
      hypre_StructMatrixNumGhost(matrix)[i] = 0;

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
    *    The stencil is a "symmetrized" version of the user's stencil
    *    as computed by hypre_StructStencilSymmetrize.
    *
    *    The `symm_elements' array is used to determine what data is
    *    explicitely stored (symm_elements[i] < 0) and what data does is
    *    not explicitely stored (symm_elements[i] >= 0), but is instead
    *    stored as the transpose coefficient at a neighboring grid point.
    *-----------------------------------------------------------------------*/

   if (hypre_StructMatrixStencil(matrix) == NULL)
   {
      user_stencil = hypre_StructMatrixUserStencil(matrix);

      hypre_StructStencilSymmetrize(user_stencil, &stencil, &symm_elements);

      stencil_shape = hypre_StructStencilShape(stencil);
      stencil_size  = hypre_StructStencilSize(stencil);

      if (!hypre_StructMatrixSymmetric(matrix))
      {
         /* store all element data */
         for (i = 0; i < stencil_size; i++)
            symm_elements[i] = -1;
         num_values = stencil_size;
      }
      else
      {
         num_values = (stencil_size + 1) / 2;
      }

      hypre_StructMatrixStencil(matrix)   = stencil;
      hypre_StructMatrixSymmElements(matrix) = symm_elements;
      hypre_StructMatrixNumValues(matrix) = num_values;
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

      data_size = 0;
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

      hypre_StructMatrixDataIndices(matrix) = data_indices;
      hypre_StructMatrixDataSize(matrix)    = data_size;
   }

   /*-----------------------------------------------------------------------
    * Set total number of nonzero coefficients
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

   hypre_BoxArray *data_boxes;
   hypre_Box      *data_box;
   hypre_Index     loop_size;
   hypre_Index     index;
   hypre_IndexRef  start;
   hypre_Index     stride;
   double         *datap;
   int             datai;
   int             i;
   int             loopi, loopj, loopk;

   hypre_StructMatrixData(matrix) = data;
   hypre_StructMatrixDataAlloced(matrix) = 0;

   /*-------------------------------------------------
    * If the matrix has a diagonal, set these entries
    * to 1 everywhere.  This reduces the complexity of
    * many computations by eliminating divide-by-zero
    * in the ghost region.
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

   data = hypre_SharedCTAlloc(double, hypre_StructMatrixDataSize(matrix));
   hypre_StructMatrixInitializeData(matrix, data);
   hypre_StructMatrixDataAlloced(matrix) = 1;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatrixSetValues
 *--------------------------------------------------------------------------*/

int 
hypre_StructMatrixSetValues( hypre_StructMatrix *matrix,
                             hypre_Index         grid_index,
                             int                 num_stencil_indices,
                             int                *stencil_indices,
                             double             *values,
                             int                 add_to              )
{
   int    ierr = 0;

   hypre_BoxArray     *boxes;
   hypre_Box          *box;

   double             *matp;

   int                 i, s;

   boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(matrix));

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
            if (add_to)
            {
               for (s = 0; s < num_stencil_indices; s++)
               {
                  matp = hypre_StructMatrixBoxDataValue(matrix, i,
                                                        stencil_indices[s],
                                                        grid_index);
                  *matp += values[s];
               }
            }
            else
            {
               for (s = 0; s < num_stencil_indices; s++)
               {
                  matp = hypre_StructMatrixBoxDataValue(matrix, i,
                                                        stencil_indices[s],
                                                        grid_index);
                  *matp = values[s];
               }
            }
         }
      }

   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_StructMatrixSetBoxValues
 *--------------------------------------------------------------------------*/

int 
hypre_StructMatrixSetBoxValues( hypre_StructMatrix *matrix,
                                hypre_Box          *value_box,
                                int                 num_stencil_indices,
                                int                *stencil_indices,
                                double             *values,
                                int                 add_to              )
{
   int    ierr = 0;

   hypre_BoxArray  *grid_boxes;
   hypre_Box       *grid_box;
   hypre_BoxArray  *box_array;
   hypre_Box       *box;
                   
   hypre_BoxArray  *data_space;
   hypre_Box       *data_box;
   hypre_IndexRef   data_start;
   hypre_Index      data_stride;
   int              datai;
   double          *datap;
                   
   hypre_Box       *dval_box;
   hypre_Index      dval_start;
   hypre_Index      dval_stride;
   int              dvali;
                   
   hypre_Index      loop_size;
                   
   int              i, s;
   int              loopi, loopj, loopk;

   /*-----------------------------------------------------------------------
    * Set up `box_array' by intersecting `box' with the grid boxes
    *-----------------------------------------------------------------------*/

   grid_boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(matrix));
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
    * Set the matrix coefficients
    *-----------------------------------------------------------------------*/

   if (box_array)
   {
      data_space = hypre_StructMatrixDataSpace(matrix);
      hypre_SetIndex(data_stride, 1, 1, 1);

      dval_box = hypre_BoxDuplicate(value_box);
      hypre_BoxIMinD(dval_box, 0) *= num_stencil_indices;
      hypre_BoxIMaxD(dval_box, 0) *= num_stencil_indices;
      hypre_BoxIMaxD(dval_box, 0) += num_stencil_indices - 1;
      hypre_SetIndex(dval_stride, num_stencil_indices, 1, 1);

      hypre_ForBoxI(i, box_array)
         {
            box      = hypre_BoxArrayBox(box_array, i);
            data_box = hypre_BoxArrayBox(data_space, i);

            /* if there was an intersection */
            if (box)
            {
               data_start = hypre_BoxIMin(box);
               hypre_CopyIndex(data_start, dval_start);
               hypre_IndexD(dval_start, 0) *= num_stencil_indices;

               for (s = 0; s < num_stencil_indices; s++)
               {
                  datap = hypre_StructMatrixBoxData(matrix, i,
                                                    stencil_indices[s]);

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

                  hypre_IndexD(dval_start, 0) ++;
               }
            }
         }

      hypre_BoxDestroy(dval_box);
   }

   hypre_BoxArrayDestroy(box_array);

   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_StructMatrixAssemble
 *--------------------------------------------------------------------------*/

int 
hypre_StructMatrixAssemble( hypre_StructMatrix *matrix )
{
   int    ierr = 0;

   int                   *num_ghost = hypre_StructMatrixNumGhost(matrix);

   hypre_BoxArrayArray   *send_boxes;
   hypre_BoxArrayArray   *recv_boxes;
   int                  **send_processes;
   int                  **recv_processes;

   hypre_Index            unit_stride;

   hypre_CommPkg         *comm_pkg;

   hypre_CommHandle      *comm_handle;

   /*-----------------------------------------------------------------------
    * If the CommPkg has not been set up, set it up
    *-----------------------------------------------------------------------*/

   comm_pkg = hypre_StructMatrixCommPkg(matrix);

   if (!comm_pkg)
   {
      hypre_SetIndex(unit_stride, 1, 1, 1);

      hypre_CreateCommInfoFromNumGhost(hypre_StructMatrixGrid(matrix),
                                       num_ghost,
                                       &send_boxes, &recv_boxes,
                                       &send_processes, &recv_processes);

      comm_pkg = hypre_CommPkgCreate(send_boxes, recv_boxes,
                                     unit_stride, unit_stride,
                                     hypre_StructMatrixDataSpace(matrix),
                                     hypre_StructMatrixDataSpace(matrix),
                                     send_processes, recv_processes,
                                     hypre_StructMatrixNumValues(matrix),
                                     hypre_StructMatrixComm(matrix),
                                     hypre_StructGridPeriodic(
                                        hypre_StructMatrixGrid(matrix)));

      hypre_StructMatrixCommPkg(matrix) = comm_pkg;
   }

   /*-----------------------------------------------------------------------
    * Update the ghost data
    *-----------------------------------------------------------------------*/

   hypre_InitializeCommunication(comm_pkg,
                                 hypre_StructMatrixData(matrix),
                                 hypre_StructMatrixData(matrix),
                                 &comm_handle);
   hypre_FinalizeCommunication(comm_handle);

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
 * hypre_StructMatrixPrint
 *--------------------------------------------------------------------------*/

int
hypre_StructMatrixPrint( char               *filename,
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

   int                   num_values;

   hypre_BoxArray       *data_space;

   int                  *symm_elements;
                   
   int                   i, j;
                   
   int                   myid;

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
   j = 0;
   for (i = 0; i < hypre_StructStencilSize(stencil); i++)
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
   hypre_PrintBoxArrayData(file, boxes, data_space, num_values,
                           hypre_StructMatrixData(matrix));

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
   hypre_BoxArrayArray   *send_boxes;
   hypre_BoxArrayArray   *recv_boxes;
   int                  **send_processes;
   int                  **recv_processes;

   hypre_Index            unit_stride;

   hypre_CommPkg         *comm_pkg;
   hypre_CommHandle      *comm_handle;

   int                    ierr = 0;

   /*------------------------------------------------------
    * Set up hypre_CommPkg
    *------------------------------------------------------*/
 
   hypre_SetIndex(unit_stride, 1, 1, 1);

   hypre_CreateCommInfoFromGrids(hypre_StructMatrixGrid(from_matrix),
                                 hypre_StructMatrixGrid(to_matrix),
                                 &send_boxes, &recv_boxes,
                                 &send_processes, &recv_processes);

   comm_pkg = hypre_CommPkgCreate(send_boxes, recv_boxes,
                                  unit_stride, unit_stride,
                                  hypre_StructMatrixDataSpace(from_matrix),
                                  hypre_StructMatrixDataSpace(to_matrix),
                                  send_processes, recv_processes,
                                  hypre_StructMatrixNumValues(from_matrix),
                                  hypre_StructMatrixComm(from_matrix),
                                  hypre_StructGridPeriodic(
                                     hypre_StructMatrixGrid(from_matrix)));
   /* is this correct for periodic? */

   /*-----------------------------------------------------------------------
    * Migrate the matrix data
    *-----------------------------------------------------------------------*/
 
   hypre_InitializeCommunication(comm_pkg,
                                 hypre_StructMatrixData(from_matrix),
                                 hypre_StructMatrixData(to_matrix),
                                 &comm_handle);
   hypre_FinalizeCommunication(comm_handle);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatrixRead
 *--------------------------------------------------------------------------*/

hypre_StructMatrix *
hypre_StructMatrixRead( MPI_Comm   comm,
                        char      *filename,
                        int       *num_ghost )
{
   FILE                 *file;
   char                  new_filename[255];
                      
   hypre_StructMatrix   *matrix;

   hypre_StructGrid     *grid;
   hypre_BoxArray       *boxes;
   int                   dim;

   hypre_StructStencil  *stencil;
   hypre_Index          *stencil_shape;
   int                   stencil_size;

   int                   num_values;

   hypre_BoxArray       *data_space;

   int                   symmetric;
                       
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

   /* read grid info */
   fscanf(file, "\nGrid:\n");
   hypre_StructGridRead(comm,file,&grid);

   /* read stencil info */
   fscanf(file, "\nStencil:\n");
   dim = hypre_StructGridDim(grid);
   fscanf(file, "%d\n", &stencil_size);
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
   hypre_StructMatrixSetNumGhost(matrix, num_ghost);
   hypre_StructMatrixInitialize(matrix);

   /*----------------------------------------
    * Read data
    *----------------------------------------*/

   boxes      = hypre_StructGridBoxes(grid);
   data_space = hypre_StructMatrixDataSpace(matrix);
   num_values = hypre_StructMatrixNumValues(matrix);
 
   fscanf(file, "\nData:\n");
   hypre_ReadBoxArrayData(file, boxes, data_space, num_values,
                          hypre_StructMatrixData(matrix));

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

