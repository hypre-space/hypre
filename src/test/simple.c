/* WM: todo - remove this file from git */

#include "_hypre_utilities.h"
#include "_hypre_utilities.hpp"
#include "HYPRE.h"
#include "_hypre_struct_mv.h"
#include "_hypre_struct_mv.hpp"

HYPRE_Int AddValuesVector( hypre_StructGrid  *gridvector,
                           hypre_StructVector *zvector,
                           HYPRE_Int          *period,
                           HYPRE_Real         value  )  ;




HYPRE_Int
my_hypre_StructVectorSetConstantValues( hypre_StructVector *vector,
                                     HYPRE_Complex       values )
{
   hypre_Box          *v_data_box;

   HYPRE_Complex      *vp;

   hypre_BoxArray     *boxes;
   hypre_Box          *box;
   hypre_Index         loop_size;
   hypre_IndexRef      start;
   hypre_Index         unit_stride;

   HYPRE_Int           i;

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   hypre_SetIndex(unit_stride, 1);

   boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(vector));
   hypre_ForBoxI(i, boxes)
   {
      box      = hypre_BoxArrayBox(boxes, i);
      start = hypre_BoxIMin(box);

      v_data_box =
         hypre_BoxArrayBox(hypre_StructVectorDataSpace(vector), i);
      vp = hypre_StructVectorBoxData(vector, i);

      hypre_BoxGetSize(box, loop_size);

#define DEVICE_VAR is_device_ptr(vp)
      hypre_BoxLoop1Begin(hypre_StructVectorNDim(vector), loop_size,
                          v_data_box, start, unit_stride, vi);
      {
         vp[vi] = values;
      }
      hypre_BoxLoop1End(vi);
#undef DEVICE_VAR
   }

   return hypre_error_flag;
}

HYPRE_Int
my_hypre_StructAxpy( HYPRE_Complex       alpha,
                     hypre_StructVector *x,
                     hypre_StructVector *y     )
{
   hypre_Box        *x_data_box;
   hypre_Box        *y_data_box;

   HYPRE_Complex    *xp;
   HYPRE_Complex    *yp;

   hypre_BoxArray   *boxes;
   hypre_Box        *box;
   hypre_Index       loop_size;
   hypre_IndexRef    start;
   hypre_Index       unit_stride;

   HYPRE_Int         i;

   hypre_SetIndex(unit_stride, 1);

   boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(y));
   hypre_ForBoxI(i, boxes)
   {
      box   = hypre_BoxArrayBox(boxes, i);
      start = hypre_BoxIMin(box);

      x_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);
      y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);

      xp = hypre_StructVectorBoxData(x, i);
      yp = hypre_StructVectorBoxData(y, i);

      hypre_BoxGetSize(box, loop_size);

/* WM: what is the DEVICE_VAR thing? */
#define DEVICE_VAR is_device_ptr(yp,xp)
      /* WM: todo */
      /* my_hypre_BoxLoop2Begin(hypre_StructVectorNDim(x), loop_size, */
      /*                     x_data_box, start, unit_stride, xi, */
      /*                     y_data_box, start, unit_stride, yi); */
      /* { */
      /*    yp[yi] += alpha * xp[xi]; */
      /* } */
      /* my_hypre_BoxLoop2End(xi, yi); */
#undef DEVICE_VAR
   }

   return hypre_error_flag;
}


/****************************
 * show device function copied from oneAPI examples
 ****************************/
#include <iomanip>
#include "dpc_common.hpp"

void ShowDevice(sycl::queue &q) {
  using namespace std;
  using namespace sycl;
  // Output platform and device information.
  auto device = q.get_device();
  auto p_name = device.get_platform().get_info<info::platform::name>();
  cout << std::setw(20) << "Platform Name: " << p_name << "\n";
  auto p_version = device.get_platform().get_info<info::platform::version>();
  cout << std::setw(20) << "Platform Version: " << p_version << "\n";
  auto d_name = device.get_info<info::device::name>();
  cout << std::setw(20) << "Device Name: " << d_name << "\n";
  auto max_work_group = device.get_info<info::device::max_work_group_size>();
  cout << std::setw(20) << "Max Work Group: " << max_work_group << "\n";
  auto max_compute_units = device.get_info<info::device::max_compute_units>();
  cout << std::setw(20) << "Max Compute Units: " << max_compute_units << "\n\n";
}

/****************************
 * main
 ****************************/

hypre_int
main( hypre_int argc,
      char *argv[] )
{

   /* initialize */
   /* hypre_MPI_Init(&argc, &argv); */
   /* HYPRE_Init(); */
   /* ShowDevice(*hypre_HandleComputeStream(hypre_handle())); */


   /* sycl::queue my_queue(sycl::default_selector{}, dpc_common::exception_handler); */
   /* ShowDevice(my_queue); */

   /* sycl::device gpu = sycl::device(sycl::cpu_selector{}); */
   /* sycl::device dev; */
   /* hypre_printf("is_host = %d\n", gpu.is_host()); */
   /* hypre_printf("is_cpu = %d\n", gpu.is_cpu()); */
   /* hypre_printf("is_cpu = %d\n", dev.is_cpu()); */
   /* hypre_printf("is_gpu = %d\n", gpu.is_gpu()); */
   /* hypre_printf("DONE\n"); */
   /* exit(0); */



   /* variables */
   HYPRE_Int           i, ix, iy, iz, ib;
   HYPRE_Int           p, q, r;
   HYPRE_Int           nx, ny, nz;
   HYPRE_Int           bx, by, bz;
   HYPRE_Int           nblocks;
   HYPRE_Int           dim;
   HYPRE_Int           sym;
   HYPRE_Int         **offsets;
   HYPRE_Int         **iupper;
   HYPRE_Int         **ilower;
   HYPRE_Int           periodic[3];
   HYPRE_Int           istart[3];
   HYPRE_StructGrid    grid;
   HYPRE_StructVector  b;
   HYPRE_StructVector  x;
   HYPRE_Int           num_ghost[6]   = {0, 0, 0, 0, 0, 0};

   dim = 2;
   sym  = 1;
   nx = 10;
   ny = 10;
   nz = 1;
   bx = 1;
   by = 1;
   bz = 1;
   p = 1;
   q = 1;
   r = 1;
   periodic[0] = 0;
   periodic[1] = 0;
   periodic[2] = 0;
   istart[0] = -3;
   istart[1] = -3;
   istart[2] = -3;

   for (i = 0; i < 2*dim; i++)
   {
      num_ghost[i]   = 1;
   }

   switch (dim)
   {
      case 1:
         nblocks = bx;
         if(sym)
         {
            offsets = hypre_CTAlloc(HYPRE_Int*,  2, HYPRE_MEMORY_HOST);
            offsets[0] = hypre_CTAlloc(HYPRE_Int,  1, HYPRE_MEMORY_HOST);
            offsets[0][0] = -1;
            offsets[1] = hypre_CTAlloc(HYPRE_Int,  1, HYPRE_MEMORY_HOST);
            offsets[1][0] = 0;
         }
         else
         {
            offsets = hypre_CTAlloc(HYPRE_Int*,  3, HYPRE_MEMORY_HOST);
            offsets[0] = hypre_CTAlloc(HYPRE_Int,  1, HYPRE_MEMORY_HOST);
            offsets[0][0] = -1;
            offsets[1] = hypre_CTAlloc(HYPRE_Int,  1, HYPRE_MEMORY_HOST);
            offsets[1][0] = 0;
            offsets[2] = hypre_CTAlloc(HYPRE_Int,  1, HYPRE_MEMORY_HOST);
            offsets[2][0] = 1;
         }
         break;

      case 2:
         nblocks = bx*by;
         if(sym)
         {
            offsets = hypre_CTAlloc(HYPRE_Int*,  3, HYPRE_MEMORY_HOST);
            offsets[0] = hypre_CTAlloc(HYPRE_Int,  2, HYPRE_MEMORY_HOST);
            offsets[0][0] = -1;
            offsets[0][1] = 0;
            offsets[1] = hypre_CTAlloc(HYPRE_Int,  2, HYPRE_MEMORY_HOST);
            offsets[1][0] = 0;
            offsets[1][1] = -1;
            offsets[2] = hypre_CTAlloc(HYPRE_Int,  2, HYPRE_MEMORY_HOST);
            offsets[2][0] = 0;
            offsets[2][1] = 0;
         }
         else
         {
            offsets = hypre_CTAlloc(HYPRE_Int*,  5, HYPRE_MEMORY_HOST);
            offsets[0] = hypre_CTAlloc(HYPRE_Int,  2, HYPRE_MEMORY_HOST);
            offsets[0][0] = -1;
            offsets[0][1] = 0;
            offsets[1] = hypre_CTAlloc(HYPRE_Int,  2, HYPRE_MEMORY_HOST);
            offsets[1][0] = 0;
            offsets[1][1] = -1;
            offsets[2] = hypre_CTAlloc(HYPRE_Int,  2, HYPRE_MEMORY_HOST);
            offsets[2][0] = 0;
            offsets[2][1] = 0;
            offsets[3] = hypre_CTAlloc(HYPRE_Int,  2, HYPRE_MEMORY_HOST);
            offsets[3][0] = 1;
            offsets[3][1] = 0;
            offsets[4] = hypre_CTAlloc(HYPRE_Int,  2, HYPRE_MEMORY_HOST);
            offsets[4][0] = 0;
            offsets[4][1] = 1;
         }
         break;

      case 3:
         nblocks = bx*by*bz;
         if(sym)
         {
            offsets = hypre_CTAlloc(HYPRE_Int*,  4, HYPRE_MEMORY_HOST);
            offsets[0] = hypre_CTAlloc(HYPRE_Int,  3, HYPRE_MEMORY_HOST);
            offsets[0][0] = -1;
            offsets[0][1] = 0;
            offsets[0][2] = 0;
            offsets[1] = hypre_CTAlloc(HYPRE_Int,  3, HYPRE_MEMORY_HOST);
            offsets[1][0] = 0;
            offsets[1][1] = -1;
            offsets[1][2] = 0;
            offsets[2] = hypre_CTAlloc(HYPRE_Int,  3, HYPRE_MEMORY_HOST);
            offsets[2][0] = 0;
            offsets[2][1] = 0;
            offsets[2][2] = -1;
            offsets[3] = hypre_CTAlloc(HYPRE_Int,  3, HYPRE_MEMORY_HOST);
            offsets[3][0] = 0;
            offsets[3][1] = 0;
            offsets[3][2] = 0;
         }
         else
         {
            offsets = hypre_CTAlloc(HYPRE_Int*,  7, HYPRE_MEMORY_HOST);
            offsets[0] = hypre_CTAlloc(HYPRE_Int,  3, HYPRE_MEMORY_HOST);
            offsets[0][0] = -1;
            offsets[0][1] = 0;
            offsets[0][2] = 0;
            offsets[1] = hypre_CTAlloc(HYPRE_Int,  3, HYPRE_MEMORY_HOST);
            offsets[1][0] = 0;
            offsets[1][1] = -1;
            offsets[1][2] = 0;
            offsets[2] = hypre_CTAlloc(HYPRE_Int,  3, HYPRE_MEMORY_HOST);
            offsets[2][0] = 0;
            offsets[2][1] = 0;
            offsets[2][2] = -1;
            offsets[3] = hypre_CTAlloc(HYPRE_Int,  3, HYPRE_MEMORY_HOST);
            offsets[3][0] = 0;
            offsets[3][1] = 0;
            offsets[3][2] = 0;
            offsets[4] = hypre_CTAlloc(HYPRE_Int,  3, HYPRE_MEMORY_HOST);
            offsets[4][0] = 1;
            offsets[4][1] = 0;
            offsets[4][2] = 0;
            offsets[5] = hypre_CTAlloc(HYPRE_Int,  3, HYPRE_MEMORY_HOST);
            offsets[5][0] = 0;
            offsets[5][1] = 1;
            offsets[5][2] = 0;
            offsets[6] = hypre_CTAlloc(HYPRE_Int,  3, HYPRE_MEMORY_HOST);
            offsets[6][0] = 0;
            offsets[6][1] = 0;
            offsets[6][2] = 1;
         }
         break;
   }



   /* initialize */
   hypre_MPI_Init(&argc, &argv);
   HYPRE_Init();

    /* prepare space for the extents */
   ilower = hypre_CTAlloc(HYPRE_Int*,  nblocks, HYPRE_MEMORY_HOST);
   iupper = hypre_CTAlloc(HYPRE_Int*,  nblocks, HYPRE_MEMORY_HOST);
   for (i = 0; i < nblocks; i++)
   {
      ilower[i] = hypre_CTAlloc(HYPRE_Int,  dim, HYPRE_MEMORY_HOST);
      iupper[i] = hypre_CTAlloc(HYPRE_Int,  dim, HYPRE_MEMORY_HOST);
   }

   /* compute ilower and iupper from (p,q,r), (bx,by,bz), and (nx,ny,nz) */
   ib = 0;
   switch (dim)
   {
      case 1:
         for (ix = 0; ix < bx; ix++)
         {
            ilower[ib][0] = istart[0]+ nx*(bx*p+ix);
            iupper[ib][0] = istart[0]+ nx*(bx*p+ix+1) - 1;
            ib++;
         }
         break;
      case 2:
         for (iy = 0; iy < by; iy++)
            for (ix = 0; ix < bx; ix++)
            {
               ilower[ib][0] = istart[0]+ nx*(bx*p+ix);
               iupper[ib][0] = istart[0]+ nx*(bx*p+ix+1) - 1;
               ilower[ib][1] = istart[1]+ ny*(by*q+iy);
               iupper[ib][1] = istart[1]+ ny*(by*q+iy+1) - 1;
               ib++;
            }
         break;
      case 3:
         for (iz = 0; iz < bz; iz++)
            for (iy = 0; iy < by; iy++)
               for (ix = 0; ix < bx; ix++)
               {
                  ilower[ib][0] = istart[0]+ nx*(bx*p+ix);
                  iupper[ib][0] = istart[0]+ nx*(bx*p+ix+1) - 1;
                  ilower[ib][1] = istart[1]+ ny*(by*q+iy);
                  iupper[ib][1] = istart[1]+ ny*(by*q+iy+1) - 1;
                  ilower[ib][2] = istart[2]+ nz*(bz*r+iz);
                  iupper[ib][2] = istart[2]+ nz*(bz*r+iz+1) - 1;
                  ib++;
               }
         break;
   }
   /* create grid */
   HYPRE_StructGridCreate(hypre_MPI_COMM_WORLD, dim, &grid);
   for (ib = 0; ib < nblocks; ib++)
   {
      /* Add to the grid a new box defined by ilower[ib], iupper[ib]...*/
      HYPRE_StructGridSetExtents(grid, ilower[ib], iupper[ib]);
   }
   HYPRE_StructGridSetPeriodic(grid, periodic);
   HYPRE_StructGridSetNumGhost(grid, num_ghost);
   HYPRE_StructGridAssemble(grid);

   /* create struct vectors */
   HYPRE_StructVectorCreate(hypre_MPI_COMM_WORLD, grid, &b);
   HYPRE_StructVectorInitialize(b);
   AddValuesVector(grid,b,periodic,1.0);
   HYPRE_StructVectorAssemble(b);

   HYPRE_StructVectorCreate(hypre_MPI_COMM_WORLD, grid, &x);
   HYPRE_StructVectorInitialize(x);
   AddValuesVector(grid,x,periodic,1.0);
   HYPRE_StructVectorAssemble(x);


   /* call set const */
   my_hypre_StructVectorSetConstantValues(x, 5.0);
   hypre_printf("my_hypre_StructVectorSetConstantValues() success!\n");

   hypre_StructVectorSetConstantValues(x, 5.0);
   hypre_printf("hypre_StructVectorSetConstantValues() success!\n");

   /* call axpy */
   /* my_hypre_StructAxpy(1.0, x, b); */







   hypre_printf("DONE\n");
   return 0;
}

HYPRE_Int
AddValuesVector( hypre_StructGrid  *gridvector,
                 hypre_StructVector *zvector,
                 HYPRE_Int          *period,
                 HYPRE_Real         value  )
{
/* #include  "_hypre_struct_mv.h" */
   HYPRE_Int ierr = 0;
   hypre_BoxArray     *gridboxes;
   HYPRE_Int          ib;
   hypre_IndexRef     ilower;
   hypre_IndexRef     iupper;
   hypre_Box          *box;
   HYPRE_Real         *values;
   HYPRE_Int          volume,dim;
#if 0 //defined(HYPRE_USING_CUDA)
   HYPRE_Int          data_location = hypre_StructGridDataLocation(hypre_StructVectorGrid(zvector));
#endif

   gridboxes =  hypre_StructGridBoxes(gridvector);
   dim       =  hypre_StructGridNDim(gridvector);

   ib=0;
   hypre_ForBoxI(ib, gridboxes)
   {
      box      = hypre_BoxArrayBox(gridboxes, ib);
      volume   =  hypre_BoxVolume(box);
#if 0 //defined(HYPRE_USING_CUDA)
      if (data_location != HYPRE_MEMORY_HOST)
      {
         values   = hypre_CTAlloc(HYPRE_Real, volume,HYPRE_MEMORY_DEVICE);
      }
      else
      {
         values   = hypre_CTAlloc(HYPRE_Real, volume,HYPRE_MEMORY_HOST);
      }
#else
      values   = hypre_CTAlloc(HYPRE_Real, volume,HYPRE_MEMORY_DEVICE);
#endif
      /*-----------------------------------------------------------
       * For periodic b.c. in all directions, need rhs to satisfy
       * compatibility condition. Achieved by setting a source and
       *  sink of equal strength.  All other problems have rhs = 1.
       *-----------------------------------------------------------*/

#define DEVICE_VAR is_device_ptr(values)
      if ((dim == 2 && period[0] != 0 && period[1] != 0) ||
          (dim == 3 && period[0] != 0 && period[1] != 0 && period[2] != 0))
      {
         hypre_LoopBegin(volume,i)
         {
            values[i] = 0.0;
            values[0]         =  value;
            values[volume - 1] = -value;

         }
         hypre_LoopEnd()
      }
      else
      {
         hypre_LoopBegin(volume,i)
         {
            values[i] = value;
         }
         hypre_LoopEnd()
      }
#undef DEVICE_VAR

      ilower = hypre_BoxIMin(box);
      iupper = hypre_BoxIMax(box);

      HYPRE_StructVectorSetBoxValues(zvector, ilower, iupper, values);

#if 0 //defined(HYPRE_USING_CUDA)
      if (data_location != HYPRE_MEMORY_HOST)
      {
          hypre_TFree(values,HYPRE_MEMORY_DEVICE);
      }
      else
      {
          hypre_TFree(values,HYPRE_MEMORY_HOST);
      }
#else
      hypre_TFree(values,HYPRE_MEMORY_DEVICE);
#endif
   }

   return ierr;
}
