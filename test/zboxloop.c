/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "_hypre_utilities.h"
#include "HYPRE_struct_ls.h"
#include "HYPRE_krylov.h"

#include "_hypre_struct_mv.h"

/*--------------------------------------------------------------------------
 * Test driver to time new boxloops and compare to the old ones
 *--------------------------------------------------------------------------*/
 
hypre_int
main( hypre_int argc,
      char *argv[] )
{
   HYPRE_Int         arg_index;
   HYPRE_Int         print_usage;
   HYPRE_Int         nx, ny, nz;
   HYPRE_Int         P, Q, R;
   HYPRE_Int         time_index;
   HYPRE_Int         num_procs, myid;
   HYPRE_Int         dim;
   HYPRE_Int         rep, reps, fail, sum;
   HYPRE_Int         size;
   hypre_Box        *x1_data_box, *x2_data_box, *x3_data_box, *x4_data_box;
   HYPRE_Int         xi1, xi2, xi3, xi4;
   HYPRE_Real       *xp1, *xp2, *xp3, *xp4;
   hypre_Index       loop_size, start, unit_stride, index;
   
   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   /* Initialize MPI */
   hypre_MPI_Init(&argc, &argv);

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
 
   dim = 3;

   nx = 10;
   ny = 10;
   nz = 10;

   P  = num_procs;
   Q  = 1;
   R  = 1;

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
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
         R  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-d") == 0 )
      {
         arg_index++;
         dim = atoi(argv[arg_index++]);
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
      hypre_printf("  -P <Px> <Py> <Pz>   : processor topology\n");
      hypre_printf("  -d <dim>            : problem dimension (2 or 3)\n");
      hypre_printf("\n");
   }

   if ( print_usage )
   {
      exit(1);
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P*Q*R) > num_procs)
   {
      if (myid == 0)
      {
         hypre_printf("Error: PxQxR is more than the number of processors\n");
      }
      exit(1);
   }
   else if ((P*Q*R) < num_procs)
   {
      if (myid == 0)
      {
         hypre_printf("Warning: PxQxR is less than the number of processors\n");
      }
   }

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_SetIndex3(start, 1, 1, 1);
   hypre_SetIndex3(loop_size, nx, ny, nz);
   hypre_SetIndex3(unit_stride, 1, 1, 1);

   x1_data_box = hypre_BoxCreate(dim);
   x2_data_box = hypre_BoxCreate(dim);
   x3_data_box = hypre_BoxCreate(dim);
   x4_data_box = hypre_BoxCreate(dim);
   hypre_SetIndex3(hypre_BoxIMin(x1_data_box), 0, 0, 0);
   hypre_SetIndex3(hypre_BoxIMax(x1_data_box), nx+1, ny+1, nz+1);
   hypre_CopyBox(x1_data_box, x2_data_box);
   hypre_CopyBox(x1_data_box, x3_data_box);
   hypre_CopyBox(x1_data_box, x4_data_box);

   size = (nx+2)*(ny+2)*(nz+2);
   xp1 = hypre_CTAlloc(HYPRE_Real, size);
   xp2 = hypre_CTAlloc(HYPRE_Real, size);
   xp3 = hypre_CTAlloc(HYPRE_Real, size);
   xp4 = hypre_CTAlloc(HYPRE_Real, size);

   reps = 1000000000/(nx*ny*nz+1000);

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      hypre_printf("Running with these driver parameters:\n");
      hypre_printf("  (nx, ny, nz)    = (%d, %d, %d)\n", nx, ny, nz);
      hypre_printf("  (Px, Py, Pz)    = (%d, %d, %d)\n", P,  Q,  R);
      hypre_printf("  dim             = %d\n", dim);
      hypre_printf("  reps            = %d\n", reps);
   }

   /*-----------------------------------------------------------
    * Check new boxloops
    *-----------------------------------------------------------*/

   /* xp1 is already initialized to 0 */

   zypre_BoxLoop1Begin(dim, loop_size,
                       x1_data_box, start, unit_stride, xi1);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(ZYPRE_BOX_PRIVATE,xi1) HYPRE_SMP_SCHEDULE
#endif
   zypre_BoxLoop1For(xi1)
   {
      xp1[xi1] ++;
   }
   zypre_BoxLoop1End(xi1);

   /* Use old boxloop to check that values are set to 1 */
   fail = 0;
   sum = 0;
   hypre_BoxLoop1Begin(3, loop_size,
                       x1_data_box, start, unit_stride, xi1);
   hypre_BoxLoop1For(xi1)
   {
      sum += xp1[xi1];
      if (xp1[xi1] != 1)
      {
         hypre_BoxLoopGetIndex(index);
         hypre_printf("*(%d,%d,%d) = %d\n",
                      index[0], index[1], index[2], (HYPRE_Int) xp1[xi1]);
         fail = 1;
      }
   }
   hypre_BoxLoop1End(xi1);

   if (sum != (nx*ny*nz))
   {
      hypre_printf("*sum = %d\n", sum);
      fail = 1;
   }
   if (fail)
   {
      exit(1);
   }

   /*-----------------------------------------------------------
    * Synchronize so that timings make sense
    *-----------------------------------------------------------*/

   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);

   /*-----------------------------------------------------------
    * Time old boxloops
    *-----------------------------------------------------------*/

   /* Time BoxLoop0 */
   time_index = hypre_InitializeTiming("BoxLoop0");
   hypre_BeginTiming(time_index);
   for (rep = 0; rep < reps; rep++)
   {
      xi1 = 0;
      hypre_BoxLoop0Begin(3, loop_size);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE) firstprivate(xi1) HYPRE_SMP_SCHEDULE
#endif
      hypre_BoxLoop0For()
      {
         xp1[xi1] += xp1[xi1];
         xi1++;
      }
      hypre_BoxLoop0End();
   }
   hypre_EndTiming(time_index);

   /* Time BoxLoop1 */
   time_index = hypre_InitializeTiming("BoxLoop1");
   hypre_BeginTiming(time_index);
   for (rep = 0; rep < reps; rep++)
   {
      hypre_BoxLoop1Begin(3, loop_size,
                          x1_data_box, start, unit_stride, xi1);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,xi1) HYPRE_SMP_SCHEDULE
#endif
      hypre_BoxLoop1For(xi1)
      {
         xp1[xi1] += xp1[xi1];
      }
      hypre_BoxLoop1End(xi1);
   }
   hypre_EndTiming(time_index);

   /* Time BoxLoop2 */
   time_index = hypre_InitializeTiming("BoxLoop2");
   hypre_BeginTiming(time_index);
   for (rep = 0; rep < reps; rep++)
   {
      hypre_BoxLoop2Begin(3, loop_size,
                          x1_data_box, start, unit_stride, xi1,
                          x2_data_box, start, unit_stride, xi2);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,xi1,xi2) HYPRE_SMP_SCHEDULE
#endif
      hypre_BoxLoop2For(xi1, xi2)
      {
         xp1[xi1] += xp1[xi1] + xp2[xi2];
      }
      hypre_BoxLoop2End(xi1, xi2);
   }
   hypre_EndTiming(time_index);

   /* Time BoxLoop3 */
   time_index = hypre_InitializeTiming("BoxLoop3");
   hypre_BeginTiming(time_index);
   for (rep = 0; rep < reps; rep++)
   {
      hypre_BoxLoop3Begin(3, loop_size,
                          x1_data_box, start, unit_stride, xi1,
                          x2_data_box, start, unit_stride, xi2,
                          x3_data_box, start, unit_stride, xi3);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,xi1,xi2,xi3) HYPRE_SMP_SCHEDULE
#endif
      hypre_BoxLoop3For(xi1, xi2, xi3)
      {
         xp1[xi1] += xp1[xi1] + xp2[xi2] + xp3[xi3];
      }
      hypre_BoxLoop3End(xi1, xi2, xi3);
   }
   hypre_EndTiming(time_index);

   /* Time BoxLoop4 */
   time_index = hypre_InitializeTiming("BoxLoop4");
   hypre_BeginTiming(time_index);
   for (rep = 0; rep < reps; rep++)
   {
      hypre_BoxLoop4Begin(3, loop_size,
                          x1_data_box, start, unit_stride, xi1,
                          x2_data_box, start, unit_stride, xi2,
                          x3_data_box, start, unit_stride, xi3,
                          x4_data_box, start, unit_stride, xi4);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,xi1,xi2,xi3,xi4) HYPRE_SMP_SCHEDULE
#endif
      hypre_BoxLoop4For(xi1, xi2, xi3, xi4)
      {
         xp1[xi1] += xp1[xi1] + xp2[xi2] + xp3[xi3] + xp4[xi4];
      }
      hypre_BoxLoop4End(xi1, xi2, xi3, xi4);
   }
   hypre_EndTiming(time_index);

   hypre_PrintTiming("Old BoxLoop times", hypre_MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

   /*-----------------------------------------------------------
    * Time new boxloops
    *-----------------------------------------------------------*/

   /* Time BoxLoop0 */
   time_index = hypre_InitializeTiming("BoxLoop0");
   hypre_BeginTiming(time_index);
   for (rep = 0; rep < reps; rep++)
   {
      xi1 = 0;
      zypre_BoxLoop0Begin(dim, loop_size);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(ZYPRE_BOX_PRIVATE) firstprivate(xi1) HYPRE_SMP_SCHEDULE
#endif
      zypre_BoxLoop0For()
      {
         xp1[xi1] += xp1[xi1];
         xi1++;
      }
      zypre_BoxLoop0End();
   }
   hypre_EndTiming(time_index);

   /* Time BoxLoop1 */
   time_index = hypre_InitializeTiming("BoxLoop1");
   hypre_BeginTiming(time_index);
   for (rep = 0; rep < reps; rep++)
   {
      zypre_BoxLoop1Begin(dim, loop_size,
                          x1_data_box, start, unit_stride, xi1);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(ZYPRE_BOX_PRIVATE,xi1) HYPRE_SMP_SCHEDULE
#endif
      zypre_BoxLoop1For(xi1)
      {
         xp1[xi1] += xp1[xi1];
      }
      zypre_BoxLoop1End(xi1);
   }
   hypre_EndTiming(time_index);

   /* Time BoxLoop2 */
   time_index = hypre_InitializeTiming("BoxLoop2");
   hypre_BeginTiming(time_index);
   for (rep = 0; rep < reps; rep++)
   {
      zypre_BoxLoop2Begin(dim, loop_size,
                          x1_data_box, start, unit_stride, xi1,
                          x2_data_box, start, unit_stride, xi2);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(ZYPRE_BOX_PRIVATE,xi1,xi2) HYPRE_SMP_SCHEDULE
#endif
      zypre_BoxLoop2For(xi1, xi2)
      {
         xp1[xi1] += xp1[xi1] + xp2[xi2];
      }
      zypre_BoxLoop2End(xi1, xi2);
   }
   hypre_EndTiming(time_index);

   /* Time BoxLoop3 */
   time_index = hypre_InitializeTiming("BoxLoop3");
   hypre_BeginTiming(time_index);
   for (rep = 0; rep < reps; rep++)
   {
      zypre_BoxLoop3Begin(dim, loop_size,
                          x1_data_box, start, unit_stride, xi1,
                          x2_data_box, start, unit_stride, xi2,
                          x3_data_box, start, unit_stride, xi3);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(ZYPRE_BOX_PRIVATE,xi1,xi2,xi3) HYPRE_SMP_SCHEDULE
#endif
      zypre_BoxLoop3For(xi1, xi2, xi3)
      {
         xp1[xi1] += xp1[xi1] + xp2[xi2] + xp3[xi3];
      }
      zypre_BoxLoop3End(xi1, xi2, xi3);
   }
   hypre_EndTiming(time_index);

   /* Time BoxLoop4 */
   time_index = hypre_InitializeTiming("BoxLoop4");
   hypre_BeginTiming(time_index);
   for (rep = 0; rep < reps; rep++)
   {
      zypre_BoxLoop4Begin(dim, loop_size,
                          x1_data_box, start, unit_stride, xi1,
                          x2_data_box, start, unit_stride, xi2,
                          x3_data_box, start, unit_stride, xi3,
                          x4_data_box, start, unit_stride, xi4);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(ZYPRE_BOX_PRIVATE,xi1,xi2,xi3,xi4) HYPRE_SMP_SCHEDULE
#endif
      zypre_BoxLoop4For(xi1, xi2, xi3, xi4)
      {
         xp1[xi1] += xp1[xi1] + xp2[xi2] + xp3[xi3] + xp4[xi4];
      }
      zypre_BoxLoop4End(xi1, xi2, xi3, xi4);
   }
   hypre_EndTiming(time_index);

   hypre_PrintTiming("New BoxLoop times", hypre_MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   hypre_BoxDestroy(x1_data_box);
   hypre_BoxDestroy(x2_data_box);
   hypre_BoxDestroy(x3_data_box);
   hypre_BoxDestroy(x4_data_box);
   hypre_TFree(xp1);
   hypre_TFree(xp2);
   hypre_TFree(xp3);
   hypre_TFree(xp4);

   /* Finalize MPI */
   hypre_MPI_Finalize();

   return (0);
}

