/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 * OpenMP Problems
 *
 * Need to fix the way these variables are set and incremented in loops:
 *   tot_nsendRowsNcols, send_ColsData_alloc, tot_sendColsData
 *
 ******************************************************************************/

#include "_hypre_sstruct_ls.h"

/*--------------------------------------------------------------------------
 * hypre_MaxwellOffProcRowCreate
 *--------------------------------------------------------------------------*/
hypre_MaxwellOffProcRow *
hypre_MaxwellOffProcRowCreate(HYPRE_Int ncols)
{
   hypre_MaxwellOffProcRow  *OffProcRow;
   HYPRE_BigInt             *cols;
   HYPRE_Real               *data;

   OffProcRow = hypre_CTAlloc(hypre_MaxwellOffProcRow,  1, HYPRE_MEMORY_HOST);
   (OffProcRow -> ncols) = ncols;

   cols = hypre_TAlloc(HYPRE_BigInt,  ncols, HYPRE_MEMORY_HOST);
   data = hypre_TAlloc(HYPRE_Real,  ncols, HYPRE_MEMORY_HOST);

   (OffProcRow -> cols) = cols;
   (OffProcRow -> data) = data;

   return OffProcRow;
}

/*--------------------------------------------------------------------------
 * hypre_MaxwellOffProcRowDestroy
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_MaxwellOffProcRowDestroy(void *OffProcRow_vdata)
{
   hypre_MaxwellOffProcRow  *OffProcRow = (hypre_MaxwellOffProcRow  *)OffProcRow_vdata;
   HYPRE_Int                 ierr = 0;

   if (OffProcRow)
   {
      hypre_TFree(OffProcRow -> cols, HYPRE_MEMORY_HOST);
      hypre_TFree(OffProcRow -> data, HYPRE_MEMORY_HOST);
   }
   hypre_TFree(OffProcRow, HYPRE_MEMORY_HOST);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructSharedDOF_ParcsrMatRowsComm
 *   Given a sstruct_grid & parcsr matrix with rows corresponding to the
 *   sstruct_grid, determine and extract the rows that must be communicated.
 *   These rows are for shared dof that geometrically lie on processor
 *   boundaries but internally are stored on one processor.
 *   Algo:
 *       for each cellbox
 *         RECVs:
 *          i)  stretch the cellbox to the variable box
 *          ii) in the appropriate (dof-dependent) direction, take the
 *              boundary and boxman_intersect to extract boxmanentries
 *              that contain these boundary edges.
 *          iii)loop over the boxmanentries and see if they belong
 *              on this proc or another proc
 *                 a) if belong on another proc, these are the recvs:
 *                    count and prepare the communication buffers and
 *                    values.
 *
 *         SENDs:
 *          i)  form layer of cells that is one layer off cellbox
 *              (stretches in the appropriate direction)
 *          ii) boxman_intersect with the cellgrid boxman
 *          iii)loop over the boxmanentries and see if they belong
 *              on this proc or another proc
 *                 a) if belong on another proc, these are the sends:
 *                    count and prepare the communication buffers and
 *                    values.
 *
 * Note: For the recv data, the dof can come from only one processor.
 *       For the send data, the dof can go to more than one processor
 *       (the same dof is on the boundary of several cells).
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_SStructSharedDOF_ParcsrMatRowsComm( hypre_SStructGrid    *grid,
                                          hypre_ParCSRMatrix   *A,
                                          HYPRE_Int            *num_offprocrows_ptr,
                                          hypre_MaxwellOffProcRow ***OffProcRows_ptr)
{
   MPI_Comm             A_comm = hypre_ParCSRMatrixComm(A);
   MPI_Comm          grid_comm = hypre_SStructGridComm(grid);

   HYPRE_Int       matrix_type = HYPRE_PARCSR;

   HYPRE_Int            nparts = hypre_SStructGridNParts(grid);
   HYPRE_Int            ndim  = hypre_SStructGridNDim(grid);

   hypre_SStructGrid     *cell_ssgrid;

   hypre_SStructPGrid    *pgrid;
   hypre_StructGrid      *cellgrid;
   hypre_BoxArray        *cellboxes;
   hypre_Box             *box, *cellbox, vbox, boxman_entry_box;

   hypre_Index            loop_size, start, lindex;
   HYPRE_BigInt           start_rank, end_rank, rank;

   HYPRE_Int              i, j, k, m, n, t, part, var, nvars;

   HYPRE_SStructVariable *vartypes;
   HYPRE_Int              nbdry_slabs = 0;
   hypre_BoxArray        *recv_slabs = NULL, *send_slabs = NULL;
   hypre_Index            varoffset;

   hypre_BoxManager     **boxmans, *cell_boxman;
   hypre_BoxManEntry    **boxman_entries, *entry;
   HYPRE_Int              nboxman_entries;

   hypre_Index            ilower, iupper, index;

   HYPRE_Int              proc, nprocs, myproc;
   HYPRE_Int             *SendToProcs, *RecvFromProcs;
   HYPRE_Int            **send_RowsNcols;       /* buffer for rows & ncols */
   HYPRE_Int             *send_RowsNcols_alloc;
   HYPRE_Int             *send_ColsData_alloc;
   HYPRE_Int             *tot_nsendRowsNcols, *tot_sendColsData;
   HYPRE_Real           **vals;  /* buffer for cols & data */

   HYPRE_BigInt          *col_inds;
   HYPRE_Real            *values;

   hypre_MPI_Request     *requests;
   hypre_MPI_Status      *status;
   HYPRE_Int            **rbuffer_RowsNcols;
   HYPRE_Real           **rbuffer_ColsData;
   HYPRE_Int              num_sends, num_recvs;

   hypre_MaxwellOffProcRow **OffProcRows;
   HYPRE_Int                *starts;

   HYPRE_Int              ierr = 0;

   hypre_BoxInit(&vbox, ndim);
   hypre_BoxInit(&boxman_entry_box, ndim);
   hypre_SetIndex(lindex, 0);

   hypre_MPI_Comm_rank(A_comm, &myproc);
   hypre_MPI_Comm_size(grid_comm, &nprocs);

   start_rank = hypre_ParCSRMatrixFirstRowIndex(A);
   end_rank  = hypre_ParCSRMatrixLastRowIndex(A);

   /* need a cellgrid boxman to determine the send boxes -> only the cell dofs
      are unique so a boxman intersect can be used to get the edges that
      must be sent. */
   HYPRE_SStructGridCreate(grid_comm, ndim, nparts, &cell_ssgrid);
   vartypes = hypre_CTAlloc(HYPRE_SStructVariable,  1, HYPRE_MEMORY_HOST);
   vartypes[0] = HYPRE_SSTRUCT_VARIABLE_CELL;

   for (i = 0; i < nparts; i++)
   {
      pgrid = hypre_SStructGridPGrid(grid, i);
      cellgrid = hypre_SStructPGridCellSGrid(pgrid);

      cellboxes = hypre_StructGridBoxes(cellgrid);
      hypre_ForBoxI(j, cellboxes)
      {
         box = hypre_BoxArrayBox(cellboxes, j);
         HYPRE_SStructGridSetExtents(cell_ssgrid, i,
                                     hypre_BoxIMin(box), hypre_BoxIMax(box));
      }
      HYPRE_SStructGridSetVariables(cell_ssgrid, i, 1, vartypes);
   }
   HYPRE_SStructGridAssemble(cell_ssgrid);
   hypre_TFree(vartypes, HYPRE_MEMORY_HOST);

   /* box algebra to determine communication */
   SendToProcs    = hypre_CTAlloc(HYPRE_Int,  nprocs, HYPRE_MEMORY_HOST);
   RecvFromProcs  = hypre_CTAlloc(HYPRE_Int,  nprocs, HYPRE_MEMORY_HOST);

   send_RowsNcols      = hypre_TAlloc(HYPRE_Int *,  nprocs, HYPRE_MEMORY_HOST);
   send_RowsNcols_alloc = hypre_TAlloc(HYPRE_Int,  nprocs, HYPRE_MEMORY_HOST);
   send_ColsData_alloc = hypre_TAlloc(HYPRE_Int,  nprocs, HYPRE_MEMORY_HOST);
   vals                = hypre_TAlloc(HYPRE_Real *,  nprocs, HYPRE_MEMORY_HOST);
   tot_nsendRowsNcols  = hypre_CTAlloc(HYPRE_Int,  nprocs, HYPRE_MEMORY_HOST);
   tot_sendColsData    = hypre_CTAlloc(HYPRE_Int,  nprocs, HYPRE_MEMORY_HOST);

   for (i = 0; i < nprocs; i++)
   {
      send_RowsNcols[i] = hypre_TAlloc(HYPRE_Int,  1000, HYPRE_MEMORY_HOST); /* initial allocation */
      send_RowsNcols_alloc[i] = 1000;

      vals[i] = hypre_TAlloc(HYPRE_Real,  2000, HYPRE_MEMORY_HOST); /* initial allocation */
      send_ColsData_alloc[i] = 2000;
   }

   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGridPGrid(grid, part);
      nvars = hypre_SStructPGridNVars(pgrid);
      vartypes = hypre_SStructPGridVarTypes(pgrid);

      cellgrid = hypre_SStructPGridCellSGrid(pgrid);
      cellboxes = hypre_StructGridBoxes(cellgrid);

      boxmans = hypre_TAlloc(hypre_BoxManager *,  nvars, HYPRE_MEMORY_HOST);
      for (t = 0; t < nvars; t++)
      {
         boxmans[t] = hypre_SStructGridBoxManager(grid, part, t);
      }
      cell_boxman = hypre_SStructGridBoxManager(cell_ssgrid, part, 0);

      hypre_ForBoxI(j, cellboxes)
      {
         cellbox = hypre_BoxArrayBox(cellboxes, j);

         for (t = 0; t < nvars; t++)
         {
            var = vartypes[t];
            hypre_SStructVariableGetOffset((hypre_SStructVariable) var,
                                           ndim, varoffset);

            /* form the variable cellbox */
            hypre_CopyBox(cellbox, &vbox);
            hypre_SubtractIndexes(hypre_BoxIMin(&vbox), varoffset, 3,
                                  hypre_BoxIMin(&vbox));

            /* boundary layer box depends on variable type */
            switch (var)
            {
               case 1:  /* node based */
               {
                  nbdry_slabs = 6;
                  recv_slabs = hypre_BoxArrayCreate(nbdry_slabs, ndim);

                  /* slab in the +/- i,j,k directions */
                  box = hypre_BoxArrayBox(recv_slabs, 0);
                  hypre_CopyBox(&vbox, box);
                  hypre_BoxIMin(box)[0] = hypre_BoxIMax(box)[0];

                  box = hypre_BoxArrayBox(recv_slabs, 1);
                  hypre_CopyBox(&vbox, box);
                  hypre_BoxIMax(box)[0] = hypre_BoxIMin(box)[0];

                  /* need to contract the slab in the i direction to avoid repeated
                     counting of some nodes. */
                  box = hypre_BoxArrayBox(recv_slabs, 2);
                  hypre_CopyBox(&vbox, box);
                  hypre_BoxIMin(box)[1] = hypre_BoxIMax(box)[1];

                  hypre_BoxIMin(box)[0]++; /* contract */
                  hypre_BoxIMax(box)[0]--; /* contract */

                  box = hypre_BoxArrayBox(recv_slabs, 3);
                  hypre_CopyBox(&vbox, box);
                  hypre_BoxIMax(box)[1] = hypre_BoxIMin(box)[1];

                  hypre_BoxIMin(box)[0]++; /* contract */
                  hypre_BoxIMax(box)[0]--; /* contract */

                  /* need to contract the slab in the i & j directions to avoid repeated
                     counting of some nodes. */
                  box = hypre_BoxArrayBox(recv_slabs, 4);
                  hypre_CopyBox(&vbox, box);
                  hypre_BoxIMin(box)[2] = hypre_BoxIMax(box)[2];

                  hypre_BoxIMin(box)[0]++; /* contract */
                  hypre_BoxIMax(box)[0]--; /* contract */
                  hypre_BoxIMin(box)[1]++; /* contract */
                  hypre_BoxIMax(box)[1]--; /* contract */

                  box = hypre_BoxArrayBox(recv_slabs, 5);
                  hypre_CopyBox(&vbox, box);
                  hypre_BoxIMax(box)[2] = hypre_BoxIMin(box)[2];

                  hypre_BoxIMin(box)[0]++; /* contract */
                  hypre_BoxIMax(box)[0]--; /* contract */
                  hypre_BoxIMin(box)[1]++; /* contract */
                  hypre_BoxIMax(box)[1]--; /* contract */

                  /* send boxes are cell-based stretching out of cellbox - i.e., cells
                     that have these edges as boundary */
                  send_slabs = hypre_BoxArrayCreate(nbdry_slabs, ndim);

                  box = hypre_BoxArrayBox(send_slabs, 0);
                  hypre_CopyBox(cellbox, box);
                  hypre_BoxIMax(box)[0]++;
                  hypre_BoxIMin(box)[0] = hypre_BoxIMax(box)[0];

                  hypre_BoxIMax(box)[2]++; /* stretch one layer +/- k*/
                  hypre_BoxIMin(box)[2]--;

                  hypre_BoxIMax(box)[1]++; /* stretch one layer +/- j*/
                  hypre_BoxIMin(box)[1]--;


                  box = hypre_BoxArrayBox(send_slabs, 1);
                  hypre_CopyBox(cellbox, box);
                  hypre_BoxIMin(box)[0]--;
                  hypre_BoxIMax(box)[0] = hypre_BoxIMin(box)[0];

                  hypre_BoxIMax(box)[2]++; /* stretch one layer +/- k*/
                  hypre_BoxIMin(box)[2]--;

                  hypre_BoxIMax(box)[1]++; /* stretch one layer +/- j*/
                  hypre_BoxIMin(box)[1]--;


                  box = hypre_BoxArrayBox(send_slabs, 2);
                  hypre_CopyBox(cellbox, box);
                  hypre_BoxIMax(box)[1]++;
                  hypre_BoxIMin(box)[1] = hypre_BoxIMax(box)[1];

                  hypre_BoxIMax(box)[2]++; /* stretch one layer +/- k*/
                  hypre_BoxIMin(box)[2]--;

                  box = hypre_BoxArrayBox(send_slabs, 3);
                  hypre_CopyBox(cellbox, box);
                  hypre_BoxIMin(box)[1]--;
                  hypre_BoxIMax(box)[1] = hypre_BoxIMin(box)[1];

                  hypre_BoxIMax(box)[2]++; /* stretch one layer +/- k*/
                  hypre_BoxIMin(box)[2]--;


                  box = hypre_BoxArrayBox(send_slabs, 4);
                  hypre_CopyBox(cellbox, box);
                  hypre_BoxIMax(box)[2]++;
                  hypre_BoxIMin(box)[2] = hypre_BoxIMax(box)[2];


                  box = hypre_BoxArrayBox(send_slabs, 5);
                  hypre_CopyBox(cellbox, box);
                  hypre_BoxIMin(box)[2]--;
                  hypre_BoxIMax(box)[2] = hypre_BoxIMin(box)[2];

                  break;
               }

               case 2:  /* x-face based */
               {
                  nbdry_slabs = 2;
                  recv_slabs = hypre_BoxArrayCreate(nbdry_slabs, ndim);

                  /* slab in the +/- i direction */
                  box = hypre_BoxArrayBox(recv_slabs, 0);
                  hypre_CopyBox(&vbox, box);
                  hypre_BoxIMin(box)[0] = hypre_BoxIMax(box)[0];

                  box = hypre_BoxArrayBox(recv_slabs, 1);
                  hypre_CopyBox(&vbox, box);
                  hypre_BoxIMax(box)[0] = hypre_BoxIMin(box)[0];

                  /* send boxes are cell-based stretching out of cellbox - i.e., cells
                     that have these edges as boundary */
                  send_slabs = hypre_BoxArrayCreate(nbdry_slabs, ndim);

                  box = hypre_BoxArrayBox(send_slabs, 0);
                  hypre_CopyBox(cellbox, box);
                  hypre_BoxIMax(box)[0]++;
                  hypre_BoxIMin(box)[0] = hypre_BoxIMax(box)[0];

                  box = hypre_BoxArrayBox(send_slabs, 1);
                  hypre_CopyBox(cellbox, box);
                  hypre_BoxIMin(box)[0]--;
                  hypre_BoxIMax(box)[0] = hypre_BoxIMin(box)[0];

                  break;
               }

               case 3:  /* y-face based */
               {
                  nbdry_slabs = 2;
                  recv_slabs = hypre_BoxArrayCreate(nbdry_slabs, ndim);

                  /* slab in the +/- j direction */
                  box = hypre_BoxArrayBox(recv_slabs, 0);
                  hypre_CopyBox(&vbox, box);
                  hypre_BoxIMin(box)[1] = hypre_BoxIMax(box)[1];

                  box = hypre_BoxArrayBox(recv_slabs, 1);
                  hypre_CopyBox(&vbox, box);
                  hypre_BoxIMax(box)[1] = hypre_BoxIMin(box)[1];

                  /* send boxes are cell-based stretching out of cellbox - i.e., cells
                     that have these edges as boundary */
                  send_slabs = hypre_BoxArrayCreate(nbdry_slabs, ndim);

                  box = hypre_BoxArrayBox(send_slabs, 0);
                  hypre_CopyBox(cellbox, box);
                  hypre_BoxIMax(box)[1]++;
                  hypre_BoxIMin(box)[1] = hypre_BoxIMax(box)[1];

                  box = hypre_BoxArrayBox(send_slabs, 1);
                  hypre_CopyBox(cellbox, box);
                  hypre_BoxIMin(box)[1]--;
                  hypre_BoxIMax(box)[1] = hypre_BoxIMin(box)[1];

                  break;
               }

               case 4:  /* z-face based */
               {
                  nbdry_slabs = 2;
                  recv_slabs = hypre_BoxArrayCreate(nbdry_slabs, ndim);

                  /* slab in the +/- k direction */
                  box = hypre_BoxArrayBox(recv_slabs, 0);
                  hypre_CopyBox(&vbox, box);
                  hypre_BoxIMin(box)[2] = hypre_BoxIMax(box)[2];

                  box = hypre_BoxArrayBox(recv_slabs, 1);
                  hypre_CopyBox(&vbox, box);
                  hypre_BoxIMax(box)[2] = hypre_BoxIMin(box)[2];

                  /* send boxes are cell-based stretching out of cellbox - i.e., cells
                     that have these edges as boundary */
                  send_slabs = hypre_BoxArrayCreate(nbdry_slabs, ndim);

                  box = hypre_BoxArrayBox(send_slabs, 0);
                  hypre_CopyBox(cellbox, box);
                  hypre_BoxIMax(box)[2]++;
                  hypre_BoxIMin(box)[2] = hypre_BoxIMax(box)[2];

                  box = hypre_BoxArrayBox(send_slabs, 1);
                  hypre_CopyBox(cellbox, box);
                  hypre_BoxIMin(box)[2]--;
                  hypre_BoxIMax(box)[2] = hypre_BoxIMin(box)[2];

                  break;
               }

               case 5:  /* x-edge based */
               {
                  nbdry_slabs = 4;
                  recv_slabs = hypre_BoxArrayCreate(nbdry_slabs, ndim);

                  /* slab in the +/- j & k direction */
                  box = hypre_BoxArrayBox(recv_slabs, 0);
                  hypre_CopyBox(&vbox, box);
                  hypre_BoxIMin(box)[1] = hypre_BoxIMax(box)[1];

                  box = hypre_BoxArrayBox(recv_slabs, 1);
                  hypre_CopyBox(&vbox, box);
                  hypre_BoxIMax(box)[1] = hypre_BoxIMin(box)[1];

                  /* need to contract the slab in the j direction to avoid repeated
                     counting of some x-edges. */
                  box = hypre_BoxArrayBox(recv_slabs, 2);
                  hypre_CopyBox(&vbox, box);
                  hypre_BoxIMin(box)[2] = hypre_BoxIMax(box)[2];

                  hypre_BoxIMin(box)[1]++; /* contract */
                  hypre_BoxIMax(box)[1]--; /* contract */

                  box = hypre_BoxArrayBox(recv_slabs, 3);
                  hypre_CopyBox(&vbox, box);
                  hypre_BoxIMax(box)[2] = hypre_BoxIMin(box)[2];

                  hypre_BoxIMin(box)[1]++; /* contract */
                  hypre_BoxIMax(box)[1]--; /* contract */

                  /* send boxes are cell-based stretching out of cellbox - i.e., cells
                     that have these edges as boundary */
                  send_slabs = hypre_BoxArrayCreate(nbdry_slabs, ndim);

                  box = hypre_BoxArrayBox(send_slabs, 0);
                  hypre_CopyBox(cellbox, box);
                  hypre_BoxIMax(box)[1]++;
                  hypre_BoxIMin(box)[1] = hypre_BoxIMax(box)[1];

                  hypre_BoxIMax(box)[2]++; /* stretch one layer +/- k*/
                  hypre_BoxIMin(box)[2]--;

                  box = hypre_BoxArrayBox(send_slabs, 1);
                  hypre_CopyBox(cellbox, box);
                  hypre_BoxIMin(box)[1]--;
                  hypre_BoxIMax(box)[1] = hypre_BoxIMin(box)[1];

                  hypre_BoxIMax(box)[2]++; /* stretch one layer +/- k*/
                  hypre_BoxIMin(box)[2]--;

                  box = hypre_BoxArrayBox(send_slabs, 2);
                  hypre_CopyBox(cellbox, box);
                  hypre_BoxIMax(box)[2]++;
                  hypre_BoxIMin(box)[2] = hypre_BoxIMax(box)[2];

                  box = hypre_BoxArrayBox(send_slabs, 3);
                  hypre_CopyBox(cellbox, box);
                  hypre_BoxIMin(box)[2]--;
                  hypre_BoxIMax(box)[2] = hypre_BoxIMin(box)[2];

                  break;
               }

               case 6:  /* y-edge based */
               {
                  nbdry_slabs = 4;
                  recv_slabs = hypre_BoxArrayCreate(nbdry_slabs, ndim);

                  /* slab in the +/- i & k direction */
                  box = hypre_BoxArrayBox(recv_slabs, 0);
                  hypre_CopyBox(&vbox, box);
                  hypre_BoxIMin(box)[0] = hypre_BoxIMax(box)[0];

                  box = hypre_BoxArrayBox(recv_slabs, 1);
                  hypre_CopyBox(&vbox, box);
                  hypre_BoxIMax(box)[0] = hypre_BoxIMin(box)[0];

                  /* need to contract the slab in the i direction to avoid repeated
                     counting of some y-edges. */
                  box = hypre_BoxArrayBox(recv_slabs, 2);
                  hypre_CopyBox(&vbox, box);
                  hypre_BoxIMin(box)[2] = hypre_BoxIMax(box)[2];

                  hypre_BoxIMin(box)[0]++; /* contract */
                  hypre_BoxIMax(box)[0]--; /* contract */

                  box = hypre_BoxArrayBox(recv_slabs, 3);
                  hypre_CopyBox(&vbox, box);
                  hypre_BoxIMax(box)[2] = hypre_BoxIMin(box)[2];

                  hypre_BoxIMin(box)[0]++; /* contract */
                  hypre_BoxIMax(box)[0]--; /* contract */

                  /* send boxes are cell-based stretching out of cellbox - i.e., cells
                     that have these edges as boundary */
                  send_slabs = hypre_BoxArrayCreate(nbdry_slabs, ndim);

                  box = hypre_BoxArrayBox(send_slabs, 0);
                  hypre_CopyBox(cellbox, box);
                  hypre_BoxIMax(box)[0]++;
                  hypre_BoxIMin(box)[0] = hypre_BoxIMax(box)[0];

                  hypre_BoxIMax(box)[2]++; /* stretch one layer +/- k*/
                  hypre_BoxIMin(box)[2]--;

                  box = hypre_BoxArrayBox(send_slabs, 1);
                  hypre_CopyBox(cellbox, box);
                  hypre_BoxIMin(box)[0]--;
                  hypre_BoxIMax(box)[0] = hypre_BoxIMin(box)[0];

                  hypre_BoxIMax(box)[2]++; /* stretch one layer +/- k*/
                  hypre_BoxIMin(box)[2]--;

                  box = hypre_BoxArrayBox(send_slabs, 2);
                  hypre_CopyBox(cellbox, box);
                  hypre_BoxIMax(box)[2]++;
                  hypre_BoxIMin(box)[2] = hypre_BoxIMax(box)[2];

                  box = hypre_BoxArrayBox(send_slabs, 3);
                  hypre_CopyBox(cellbox, box);
                  hypre_BoxIMin(box)[2]--;
                  hypre_BoxIMax(box)[2] = hypre_BoxIMin(box)[2];

                  break;
               }

               case 7:  /* z-edge based */
               {
                  nbdry_slabs = 4;
                  recv_slabs = hypre_BoxArrayCreate(nbdry_slabs, ndim);

                  /* slab in the +/- i & j direction */
                  box = hypre_BoxArrayBox(recv_slabs, 0);
                  hypre_CopyBox(&vbox, box);
                  hypre_BoxIMin(box)[0] = hypre_BoxIMax(box)[0];

                  box = hypre_BoxArrayBox(recv_slabs, 1);
                  hypre_CopyBox(&vbox, box);
                  hypre_BoxIMax(box)[0] = hypre_BoxIMin(box)[0];

                  /* need to contract the slab in the i direction to avoid repeated
                     counting of some z-edges. */
                  box = hypre_BoxArrayBox(recv_slabs, 2);
                  hypre_CopyBox(&vbox, box);
                  hypre_BoxIMin(box)[1] = hypre_BoxIMax(box)[1];

                  hypre_BoxIMin(box)[0]++; /* contract */
                  hypre_BoxIMax(box)[0]--; /* contract */

                  box = hypre_BoxArrayBox(recv_slabs, 3);
                  hypre_CopyBox(&vbox, box);
                  hypre_BoxIMax(box)[1] = hypre_BoxIMin(box)[1];

                  hypre_BoxIMin(box)[0]++; /* contract */
                  hypre_BoxIMax(box)[0]--; /* contract */

                  /* send boxes are cell-based stretching out of cellbox - i.e., cells
                     that have these edges as boundary */
                  send_slabs = hypre_BoxArrayCreate(nbdry_slabs, ndim);

                  box = hypre_BoxArrayBox(send_slabs, 0);
                  hypre_CopyBox(cellbox, box);
                  hypre_BoxIMax(box)[1]++;
                  hypre_BoxIMin(box)[1] = hypre_BoxIMax(box)[1];

                  hypre_BoxIMax(box)[0]++; /* stretch one layer +/- i*/
                  hypre_BoxIMin(box)[0]--;

                  box = hypre_BoxArrayBox(send_slabs, 1);
                  hypre_CopyBox(cellbox, box);
                  hypre_BoxIMin(box)[1]--;
                  hypre_BoxIMax(box)[1] = hypre_BoxIMin(box)[1];

                  hypre_BoxIMax(box)[0]++; /* stretch one layer +/- i*/
                  hypre_BoxIMin(box)[0]--;

                  box = hypre_BoxArrayBox(send_slabs, 2);
                  hypre_CopyBox(cellbox, box);
                  hypre_BoxIMax(box)[0]++;
                  hypre_BoxIMin(box)[0] = hypre_BoxIMax(box)[0];

                  box = hypre_BoxArrayBox(send_slabs, 3);
                  hypre_CopyBox(cellbox, box);
                  hypre_BoxIMin(box)[0]--;
                  hypre_BoxIMax(box)[0] = hypre_BoxIMin(box)[0];

                  break;
               }

            }  /* switch(var) */

            /* determine no. of recv rows */
            for (i = 0; i < nbdry_slabs; i++)
            {
               box = hypre_BoxArrayBox(recv_slabs, i);
               hypre_BoxManIntersect(boxmans[t], hypre_BoxIMin(box), hypre_BoxIMax(box),
                                     &boxman_entries, &nboxman_entries);

               for (m = 0; m < nboxman_entries; m++)
               {
                  hypre_SStructBoxManEntryGetProcess(boxman_entries[m], &proc);
                  if (proc != myproc)
                  {
                     hypre_BoxManEntryGetExtents(boxman_entries[m], ilower, iupper);
                     hypre_BoxSetExtents(&boxman_entry_box, ilower, iupper);
                     hypre_IntersectBoxes(&boxman_entry_box, box, &boxman_entry_box);

                     RecvFromProcs[proc] += hypre_BoxVolume(&boxman_entry_box);
                  }
               }
               hypre_TFree(boxman_entries, HYPRE_MEMORY_HOST);

               /* determine send rows. Note the cell_boxman */
               box = hypre_BoxArrayBox(send_slabs, i);
               hypre_BoxManIntersect(cell_boxman, hypre_BoxIMin(box), hypre_BoxIMax(box),
                                     &boxman_entries, &nboxman_entries);

               for (m = 0; m < nboxman_entries; m++)
               {
                  hypre_SStructBoxManEntryGetProcess(boxman_entries[m], &proc);
                  if (proc != myproc)
                  {
                     hypre_BoxManEntryGetExtents(boxman_entries[m], ilower, iupper);
                     hypre_BoxSetExtents(&boxman_entry_box, ilower, iupper);
                     hypre_IntersectBoxes(&boxman_entry_box, box, &boxman_entry_box);

                     /* not correct box piece right now. Need to determine
                        the correct var box - extend to var_box and then intersect
                        with vbox */
                     hypre_SubtractIndexes(hypre_BoxIMin(&boxman_entry_box),
                                           varoffset, 3,
                                           hypre_BoxIMin(&boxman_entry_box));
                     hypre_IntersectBoxes(&boxman_entry_box, &vbox, &boxman_entry_box);

                     SendToProcs[proc] += 2 * hypre_BoxVolume(&boxman_entry_box);
                     /* check to see if sufficient memory allocation for send_rows */
                     if (SendToProcs[proc] > send_RowsNcols_alloc[proc])
                     {
                        send_RowsNcols_alloc[proc] = SendToProcs[proc];
                        send_RowsNcols[proc] =
                           hypre_TReAlloc(send_RowsNcols[proc],  HYPRE_Int,
                                          send_RowsNcols_alloc[proc], HYPRE_MEMORY_HOST);
                     }

                     hypre_BoxGetSize(&boxman_entry_box, loop_size);
                     hypre_CopyIndex(hypre_BoxIMin(&boxman_entry_box), start);

                     hypre_SerialBoxLoop0Begin(ndim, loop_size);
                     {
                        zypre_BoxLoopGetIndex(lindex);
                        hypre_SetIndex3(index, lindex[0], lindex[1], lindex[2]);
                        hypre_AddIndexes(index, start, 3, index);

                        hypre_SStructGridFindBoxManEntry(grid, part, index, t,
                                                         &entry);
                        if (entry)
                        {
                           hypre_SStructBoxManEntryGetGlobalRank(entry, index,
                                                                 &rank, matrix_type);

                           /* index may still be off myproc because vbox was formed
                              by expanding the cellbox to the variable box without
                              checking (difficult) the whole expanded box is on myproc */
                           if (rank <= end_rank && rank >= start_rank)
                           {
                              send_RowsNcols[proc][tot_nsendRowsNcols[proc]] = rank;
                              tot_nsendRowsNcols[proc]++;

                              HYPRE_ParCSRMatrixGetRow((HYPRE_ParCSRMatrix) A, rank, &n,
                                                       &col_inds, &values);
                              send_RowsNcols[proc][tot_nsendRowsNcols[proc]] = n;
                              tot_nsendRowsNcols[proc]++;

                              /* check if sufficient memory allocation in the data arrays */
                              if ( (tot_sendColsData[proc] + 2 * n) > send_ColsData_alloc[proc] )
                              {
                                 send_ColsData_alloc[proc] += 2000;
                                 vals[proc] = hypre_TReAlloc(vals[proc],  HYPRE_Real,
                                                             send_ColsData_alloc[proc], HYPRE_MEMORY_HOST);
                              }
                              for (k = 0; k < n; k++)
                              {
                                 vals[proc][tot_sendColsData[proc]] = (HYPRE_Real) col_inds[k];
                                 tot_sendColsData[proc]++;
                                 vals[proc][tot_sendColsData[proc]] = values[k];
                                 tot_sendColsData[proc]++;
                              }
                              HYPRE_ParCSRMatrixRestoreRow((HYPRE_ParCSRMatrix) A, rank, &n,
                                                           &col_inds, &values);
                           }  /* if (rank <= end_rank && rank >= start_rank) */
                        }     /* if (entry) */
                     }
                     hypre_SerialBoxLoop0End();

                  }  /* if (proc != myproc) */
               }     /* for (m= 0; m< nboxman_entries; m++) */
               hypre_TFree(boxman_entries, HYPRE_MEMORY_HOST);

            }  /* for (i= 0; i< nbdry_slabs; i++) */
            hypre_BoxArrayDestroy(send_slabs);
            hypre_BoxArrayDestroy(recv_slabs);

         }  /* for (t= 0; t< nvars; t++) */
      }     /* hypre_ForBoxI(j, cellboxes) */
      hypre_TFree(boxmans, HYPRE_MEMORY_HOST);
   }  /* for (part= 0; part< nparts; part++) */

   HYPRE_SStructGridDestroy(cell_ssgrid);

   num_sends = 0;
   num_recvs = 0;
   k = 0;
   starts = hypre_CTAlloc(HYPRE_Int,  nprocs + 1, HYPRE_MEMORY_HOST);
   for (i = 0; i < nprocs; i++)
   {
      starts[i + 1] = starts[i] + RecvFromProcs[i];
      if (RecvFromProcs[i])
      {
         num_recvs++;
         k += RecvFromProcs[i];
      }

      if (tot_sendColsData[i])
      {
         num_sends++;
      }
   }
   OffProcRows = hypre_TAlloc(hypre_MaxwellOffProcRow *,  k, HYPRE_MEMORY_HOST);
   *num_offprocrows_ptr = k;

   requests = hypre_CTAlloc(hypre_MPI_Request,  num_sends + num_recvs, HYPRE_MEMORY_HOST);
   status  = hypre_CTAlloc(hypre_MPI_Status,  num_sends + num_recvs, HYPRE_MEMORY_HOST);

   /* send row size data */
   j = 0;
   rbuffer_RowsNcols = hypre_TAlloc(HYPRE_Int *,  nprocs, HYPRE_MEMORY_HOST);
   rbuffer_ColsData = hypre_TAlloc(HYPRE_Real *,  nprocs, HYPRE_MEMORY_HOST);

   for (proc = 0; proc < nprocs; proc++)
   {
      if (RecvFromProcs[proc])
      {
         rbuffer_RowsNcols[proc] = hypre_TAlloc(HYPRE_Int,  2 * RecvFromProcs[proc], HYPRE_MEMORY_HOST);
         hypre_MPI_Irecv(rbuffer_RowsNcols[proc], 2 * RecvFromProcs[proc], HYPRE_MPI_INT,
                         proc, 0, grid_comm, &requests[j++]);
      }  /* if (RecvFromProcs[proc]) */

   }     /* for (proc= 0; proc< nprocs; proc++) */

   for (proc = 0; proc < nprocs; proc++)
   {
      if (tot_nsendRowsNcols[proc])
      {
         hypre_MPI_Isend(send_RowsNcols[proc], tot_nsendRowsNcols[proc], HYPRE_MPI_INT, proc,
                         0, grid_comm, &requests[j++]);
      }
   }

   hypre_MPI_Waitall(j, requests, status);

   /* unpack data */
   for (proc = 0; proc < nprocs; proc++)
   {
      send_RowsNcols_alloc[proc] = 0;
      if (RecvFromProcs[proc])
      {
         m = 0; ;
         for (i = 0; i < RecvFromProcs[proc]; i++)
         {
            /* rbuffer_RowsNcols[m] has the row & rbuffer_RowsNcols[m+1] the col size */
            OffProcRows[starts[proc] + i] = hypre_MaxwellOffProcRowCreate(rbuffer_RowsNcols[proc][m + 1]);
            (OffProcRows[starts[proc] + i] -> row)  = rbuffer_RowsNcols[proc][m];
            (OffProcRows[starts[proc] + i] -> ncols) = rbuffer_RowsNcols[proc][m + 1];

            send_RowsNcols_alloc[proc] += rbuffer_RowsNcols[proc][m + 1];
            m += 2;
         }

         rbuffer_ColsData[proc] = hypre_TAlloc(HYPRE_Real,  2 * send_RowsNcols_alloc[proc],
                                               HYPRE_MEMORY_HOST);
         hypre_TFree(rbuffer_RowsNcols[proc], HYPRE_MEMORY_HOST);
      }
   }

   hypre_TFree(rbuffer_RowsNcols, HYPRE_MEMORY_HOST);
   hypre_TFree(requests, HYPRE_MEMORY_HOST);
   hypre_TFree(status, HYPRE_MEMORY_HOST);

   requests = hypre_CTAlloc(hypre_MPI_Request,  num_sends + num_recvs, HYPRE_MEMORY_HOST);
   status  = hypre_CTAlloc(hypre_MPI_Status,  num_sends + num_recvs, HYPRE_MEMORY_HOST);

   /* send row data */
   j = 0;
   for (proc = 0; proc < nprocs; proc++)
   {
      if (RecvFromProcs[proc])
      {
         hypre_MPI_Irecv(rbuffer_ColsData[proc], 2 * send_RowsNcols_alloc[proc], HYPRE_MPI_REAL,
                         proc, 1, grid_comm, &requests[j++]);
      }  /* if (RecvFromProcs[proc]) */
   }     /* for (proc= 0; proc< nprocs; proc++) */

   for (proc = 0; proc < nprocs; proc++)
   {
      if (tot_sendColsData[proc])
      {
         hypre_MPI_Isend(vals[proc], tot_sendColsData[proc], HYPRE_MPI_REAL, proc,
                         1, grid_comm, &requests[j++]);
      }
   }

   hypre_MPI_Waitall(j, requests, status);

   /* unpack data */
   for (proc = 0; proc < nprocs; proc++)
   {
      if (RecvFromProcs[proc])
      {
         k = 0;
         for (i = 0; i < RecvFromProcs[proc]; i++)
         {
            col_inds = (OffProcRows[starts[proc] + i] -> cols);
            values  = (OffProcRows[starts[proc] + i] -> data);
            m       = (OffProcRows[starts[proc] + i] -> ncols);

            for (t = 0; t < m; t++)
            {
               col_inds[t] = (HYPRE_Int) rbuffer_ColsData[proc][k++];
               values[t]  = rbuffer_ColsData[proc][k++];
            }
         }
         hypre_TFree(rbuffer_ColsData[proc], HYPRE_MEMORY_HOST);
      }  /* if (RecvFromProcs[proc]) */

   }     /* for (proc= 0; proc< nprocs; proc++) */
   hypre_TFree(rbuffer_ColsData, HYPRE_MEMORY_HOST);

   hypre_TFree(requests, HYPRE_MEMORY_HOST);
   hypre_TFree(status, HYPRE_MEMORY_HOST);
   for (proc = 0; proc < nprocs; proc++)
   {
      hypre_TFree(send_RowsNcols[proc], HYPRE_MEMORY_HOST);
      hypre_TFree(vals[proc], HYPRE_MEMORY_HOST);
   }
   hypre_TFree(send_RowsNcols, HYPRE_MEMORY_HOST);
   hypre_TFree(vals, HYPRE_MEMORY_HOST);
   hypre_TFree(tot_sendColsData, HYPRE_MEMORY_HOST);
   hypre_TFree(tot_nsendRowsNcols, HYPRE_MEMORY_HOST);
   hypre_TFree(send_ColsData_alloc, HYPRE_MEMORY_HOST);
   hypre_TFree(send_RowsNcols_alloc, HYPRE_MEMORY_HOST);
   hypre_TFree(SendToProcs, HYPRE_MEMORY_HOST);
   hypre_TFree(RecvFromProcs, HYPRE_MEMORY_HOST);
   hypre_TFree(starts, HYPRE_MEMORY_HOST);

   *OffProcRows_ptr = OffProcRows;

   return ierr;
}
