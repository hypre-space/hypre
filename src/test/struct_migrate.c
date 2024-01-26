/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "_hypre_utilities.h"
#include "HYPRE_struct_mv.h"
/* RDF: This include is only needed for AddValuesVector() */
#include "_hypre_struct_mv.h"

HYPRE_Int AddValuesVector( hypre_StructGrid   *grid,
                           hypre_StructVector *vector,
                           HYPRE_Real          value );

/*--------------------------------------------------------------------------
 * Test driver for structured matrix interface (structured storage)
 *--------------------------------------------------------------------------*/

/*----------------------------------------------------------------------
 * Standard 7-point laplacian in 3D with grid and anisotropy determined
 * as command line arguments.  Do `driver -help' for usage info.
 *----------------------------------------------------------------------*/

hypre_int
main( hypre_int argc,
      char *argv[] )
{
   HYPRE_Int           arg_index;
   HYPRE_Int           print_usage;
   HYPRE_Int           nx, ny, nz;
   HYPRE_Int           P, Q, R;
   HYPRE_Int           bx, by, bz;

   HYPRE_StructGrid    from_grid, to_grid;
   HYPRE_StructVector  from_vector, to_vector, check_vector;
   HYPRE_CommPkg       comm_pkg;

   HYPRE_Int           time_index;
   HYPRE_Int           num_procs, myid;

   HYPRE_Int           p, q, r;
   HYPRE_Int           dim;
   HYPRE_Int           nblocks = 0;
   HYPRE_Int         **ilower, **iupper, **iupper2;
   HYPRE_Int           istart[3];
   HYPRE_Int           i, ix, iy, iz, ib;
   HYPRE_Int           print_system = 0;

   HYPRE_Real          check;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   /* Initialize MPI */
   hypre_MPI_Init(&argc, &argv);
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------------
    * GPU Device binding
    * Must be done before HYPRE_Initialize() and should not be changed after
    *-----------------------------------------------------------------*/
   hypre_bind_device_id(-1, myid, num_procs, hypre_MPI_COMM_WORLD);

   /*-----------------------------------------------------------
    * Initialize : must be the first HYPRE function to call
    *-----------------------------------------------------------*/
   HYPRE_Initialize();
   HYPRE_DeviceInitialize();

#if defined(HYPRE_USING_KOKKOS)
   Kokkos::initialize (argc, argv);
#endif

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   dim = 3;

   nx = 2;
   ny = 2;
   nz = 2;

   P  = num_procs;
   Q  = 1;
   R  = 1;

   p = q = r = 1;

   bx = 1;
   by = 1;
   bz = 1;

   istart[0] = 1;
   istart[1] = 1;
   istart[2] = 1;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   print_usage = 0;
   arg_index = 1;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
         nz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-istart") == 0 )
      {
         arg_index++;
         istart[0] = atoi(argv[arg_index++]);
         istart[1] = atoi(argv[arg_index++]);
         istart[2] = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
         R  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-b") == 0 )
      {
         arg_index++;
         bx = atoi(argv[arg_index++]);
         by = atoi(argv[arg_index++]);
         bz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-d") == 0 )
      {
         arg_index++;
         dim = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-print") == 0 )
      {
         arg_index++;
         print_system = 1;
      }
      else if ( strcmp(argv[arg_index], "-help") == 0 )
      {
         print_usage = 1;
         break;
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Print usage info
    *-----------------------------------------------------------*/

   if ( (print_usage) && (myid == 0) )
   {
      hypre_printf("\n");
      hypre_printf("Usage: %s [<options>]\n", argv[0]);
      hypre_printf("\n");
      hypre_printf("  -n <nx> <ny> <nz>   : problem size per block\n");
      hypre_printf("  -istart <ix> <iy> <iz> : start of box\n");
      hypre_printf("  -P <Px> <Py> <Pz>   : processor topology\n");
      hypre_printf("  -b <bx> <by> <bz>   : blocking per processor\n");
      hypre_printf("  -d <dim>            : problem dimension (2 or 3)\n");
      hypre_printf("  -print              : print vectors\n");
      hypre_printf("\n");
   }

   if ( print_usage )
   {
      exit(1);
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P * Q * R) > num_procs)
   {
      if (myid == 0)
      {
         hypre_printf("Error: PxQxR is more than the number of processors\n");
      }
      exit(1);
   }
   else if ((P * Q * R) < num_procs)
   {
      if (myid == 0)
      {
         hypre_printf("Warning: PxQxR is less than the number of processors\n");
      }
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      hypre_printf("Running with these driver parameters:\n");
      hypre_printf("  (nx, ny, nz)    = (%d, %d, %d)\n", nx, ny, nz);
      hypre_printf("  (ix, iy, iz)    = (%d, %d, %d)\n",
                   istart[0], istart[1], istart[2]);
      hypre_printf("  (Px, Py, Pz)    = (%d, %d, %d)\n", P,  Q,  R);
      hypre_printf("  (bx, by, bz)    = (%d, %d, %d)\n", bx, by, bz);
      hypre_printf("  dim             = %d\n", dim);
   }

   /*-----------------------------------------------------------
    * Set up the stencil structure (7 points) when matrix is NOT read from file
    * Set up the grid structure used when NO files are read
    *-----------------------------------------------------------*/

   switch (dim)
   {
      case 1:
         nblocks = bx;
         p = myid % P;
         break;

      case 2:
         nblocks = bx * by;
         p = myid % P;
         q = (( myid - p) / P) % Q;
         break;

      case 3:
         nblocks = bx * by * bz;
         p = myid % P;
         q = (( myid - p) / P) % Q;
         r = ( myid - p - P * q) / ( P * Q );
         break;
   }

   if (myid >= (P * Q * R))
   {
      /* My processor has no data on it */
      nblocks = bx = by = bz = 0;
   }

   /*-----------------------------------------------------------
    * prepare space for the extents
    *-----------------------------------------------------------*/

   ilower = hypre_CTAlloc(HYPRE_Int*,  nblocks, HYPRE_MEMORY_HOST);
   iupper = hypre_CTAlloc(HYPRE_Int*,  nblocks, HYPRE_MEMORY_HOST);
   iupper2 = hypre_CTAlloc(HYPRE_Int*,  nblocks, HYPRE_MEMORY_HOST);
   for (i = 0; i < nblocks; i++)
   {
      ilower[i] = hypre_CTAlloc(HYPRE_Int,  dim, HYPRE_MEMORY_HOST);
      iupper[i] = hypre_CTAlloc(HYPRE_Int,  dim, HYPRE_MEMORY_HOST);
      iupper2[i] = hypre_CTAlloc(HYPRE_Int,  dim, HYPRE_MEMORY_HOST);
   }

   ib = 0;
   switch (dim)
   {
      case 1:
         for (ix = 0; ix < bx; ix++)
         {
            ilower[ib][0] = istart[0] + nx * (bx * p + ix);
            iupper[ib][0] = istart[0] + nx * (bx * p + ix + 1) - 1;
            iupper2[ib][0] = iupper[ib][0];
            if ( (ix == (bx - 1)) && (p < (P - 1)) )
            {
               iupper2[ib][0] = iupper[ib][0] + 1;
            }
            ib++;
         }
         break;
      case 2:
         for (iy = 0; iy < by; iy++)
            for (ix = 0; ix < bx; ix++)
            {
               ilower[ib][0] = istart[0] + nx * (bx * p + ix);
               iupper[ib][0] = istart[0] + nx * (bx * p + ix + 1) - 1;
               ilower[ib][1] = istart[1] + ny * (by * q + iy);
               iupper[ib][1] = istart[1] + ny * (by * q + iy + 1) - 1;
               iupper2[ib][0] = iupper[ib][0];
               iupper2[ib][1] = iupper[ib][1];
               if ( (ix == (bx - 1)) && (p < (P - 1)) )
               {
                  iupper2[ib][0] = iupper[ib][0] + 1;
               }
               if ( (iy == (by - 1)) && (q < (Q - 1)) )
               {
                  iupper2[ib][1] = iupper[ib][1] + 1;
               }
               ib++;
            }
         break;
      case 3:
         for (iz = 0; iz < bz; iz++)
            for (iy = 0; iy < by; iy++)
               for (ix = 0; ix < bx; ix++)
               {
                  ilower[ib][0] = istart[0] + nx * (bx * p + ix);
                  iupper[ib][0] = istart[0] + nx * (bx * p + ix + 1) - 1;
                  ilower[ib][1] = istart[1] + ny * (by * q + iy);
                  iupper[ib][1] = istart[1] + ny * (by * q + iy + 1) - 1;
                  ilower[ib][2] = istart[2] + nz * (bz * r + iz);
                  iupper[ib][2] = istart[2] + nz * (bz * r + iz + 1) - 1;
                  iupper2[ib][0] = iupper[ib][0];
                  iupper2[ib][1] = iupper[ib][1];
                  iupper2[ib][2] = iupper[ib][2];
                  if ( (ix == (bx - 1)) && (p < (P - 1)) )
                  {
                     iupper2[ib][0] = iupper[ib][0] + 1;
                  }
                  if ( (iy == (by - 1)) && (q < (Q - 1)) )
                  {
                     iupper2[ib][1] = iupper[ib][1] + 1;
                  }
                  if ( (iz == (bz - 1)) && (r < (R - 1)) )
                  {
                     iupper2[ib][2] = iupper[ib][2] + 1;
                  }
                  ib++;
               }
         break;
   }

   HYPRE_StructGridCreate(hypre_MPI_COMM_WORLD, dim, &from_grid);
   HYPRE_StructGridCreate(hypre_MPI_COMM_WORLD, dim, &to_grid);
   for (ib = 0; ib < nblocks; ib++)
   {
      HYPRE_StructGridSetExtents(from_grid, ilower[ib], iupper[ib]);
      HYPRE_StructGridSetExtents(to_grid, ilower[ib], iupper2[ib]);
   }
   HYPRE_StructGridAssemble(from_grid);
   HYPRE_StructGridAssemble(to_grid);

   /*-----------------------------------------------------------
    * Set up the vectors
    *-----------------------------------------------------------*/

   HYPRE_StructVectorCreate(hypre_MPI_COMM_WORLD, from_grid, &from_vector);
   HYPRE_StructVectorInitialize(from_vector);
   AddValuesVector(from_grid, from_vector, 1.0);
   HYPRE_StructVectorAssemble(from_vector);

   HYPRE_StructVectorCreate(hypre_MPI_COMM_WORLD, to_grid, &to_vector);
   HYPRE_StructVectorInitialize(to_vector);
   AddValuesVector(to_grid, to_vector, 0.0);
   HYPRE_StructVectorAssemble(to_vector);

   /* Vector used to check the migration */
   HYPRE_StructVectorCreate(hypre_MPI_COMM_WORLD, to_grid, &check_vector);
   HYPRE_StructVectorInitialize(check_vector);
   AddValuesVector(to_grid, check_vector, 1.0);
   HYPRE_StructVectorAssemble(check_vector);

   /*-----------------------------------------------------------
    * Migrate
    *-----------------------------------------------------------*/

   time_index = hypre_InitializeTiming("Struct Migrate");
   hypre_BeginTiming(time_index);

   HYPRE_StructVectorGetMigrateCommPkg(from_vector, to_vector, &comm_pkg);
   HYPRE_StructVectorMigrate(comm_pkg, from_vector, to_vector);
   HYPRE_CommPkgDestroy(comm_pkg);

   hypre_EndTiming(time_index);
   hypre_PrintTiming("Struct Migrate", hypre_MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);

   /*-----------------------------------------------------------
    * Check the migration and print the result
    *-----------------------------------------------------------*/

   hypre_StructAxpy(-1.0, to_vector, check_vector);
   check = hypre_StructInnerProd (check_vector, check_vector);

   if (myid == 0)
   {
      hypre_printf("\nCheck = %1.0f (success = 0)\n\n", check);
   }

   /*-----------------------------------------------------------
    * Print out the vectors
    *-----------------------------------------------------------*/

   if (print_system)
   {
      HYPRE_StructVectorPrint("struct_migrate.out.xfr", from_vector, 0);
      HYPRE_StructVectorPrint("struct_migrate.out.xto", to_vector, 0);
   }

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   HYPRE_StructGridDestroy(from_grid);
   HYPRE_StructGridDestroy(to_grid);

   for (i = 0; i < nblocks; i++)
   {
      hypre_TFree(ilower[i], HYPRE_MEMORY_HOST);
      hypre_TFree(iupper[i], HYPRE_MEMORY_HOST);
      hypre_TFree(iupper2[i], HYPRE_MEMORY_HOST);
   }
   hypre_TFree(ilower, HYPRE_MEMORY_HOST);
   hypre_TFree(iupper, HYPRE_MEMORY_HOST);
   hypre_TFree(iupper2, HYPRE_MEMORY_HOST);

   HYPRE_StructVectorDestroy(from_vector);
   HYPRE_StructVectorDestroy(to_vector);
   HYPRE_StructVectorDestroy(check_vector);

#if defined(HYPRE_USING_KOKKOS)
   Kokkos::finalize ();
#endif

   /* Finalize Hypre */
   HYPRE_Finalize();

   /* Finalize MPI */
   hypre_MPI_Finalize();

   return (0);
}

/*-------------------------------------------------------------------------
 * Add constant values to a vector.
 *-------------------------------------------------------------------------*/

HYPRE_Int
AddValuesVector( hypre_StructGrid   *grid,
                 hypre_StructVector *vector,
                 HYPRE_Real          value )
{
   HYPRE_Int          i, ierr = 0;
   hypre_BoxArray    *gridboxes;
   HYPRE_Int          ib;
   hypre_IndexRef     ilower;
   hypre_IndexRef     iupper;
   hypre_Box         *box;
   HYPRE_Real        *values;
   HYPRE_Real        *values_h;
   HYPRE_Int          volume;

   HYPRE_MemoryLocation memory_location = hypre_StructVectorMemoryLocation(vector);

   gridboxes = hypre_StructGridBoxes(grid);

   ib = 0;
   hypre_ForBoxI(ib, gridboxes)
   {
      box    = hypre_BoxArrayBox(gridboxes, ib);
      volume = hypre_BoxVolume(box);
      values = hypre_CTAlloc(HYPRE_Real, volume, memory_location);
      values_h = hypre_CTAlloc(HYPRE_Real, volume, HYPRE_MEMORY_HOST);

      for (i = 0; i < volume; i++)
      {
         values_h[i] = value;
      }

      hypre_TMemcpy(values, values_h, HYPRE_Real, volume, memory_location, HYPRE_MEMORY_HOST);

      ilower = hypre_BoxIMin(box);
      iupper = hypre_BoxIMax(box);
      HYPRE_StructVectorSetBoxValues(vector, ilower, iupper, values);
      hypre_TFree(values, memory_location);
      hypre_TFree(values_h, HYPRE_MEMORY_HOST);
   }

   return ierr;
}
