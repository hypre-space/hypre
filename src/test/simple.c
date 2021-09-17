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



/*********************************************************************
 * put this in _hypre_utilities.hpp ? 
 * WM: todo - if you can wrap the basic parallel_for call for use elsewhere...
 *********************************************************************/
/* #define HYPRE_SYCL_1D_LAUNCH(kernel_name, gridsize, blocksize, ...)                                                  \ */
/* {                                                                                                                    \ */
/*    if ( gridsize[0] == 0 || blocksize[0] == 0 )                                                                      \ */
/*    {                                                                                                                 \ */
/*       hypre_printf("Error %s %d: Invalid SYCL 1D launch parameters grid/block (%d) (%d)\n",                          \ */
/*                    __FILE__, __LINE__,                                                                               \ */
/*                    gridsize[0], blocksize[0]);                                                                       \ */
/*       assert(0); exit(1);                                                                                            \ */
/*    }                                                                                                                 \ */
/*    else                                                                                                              \ */
/*    {                                                                                                                 \ */
/*       hypre_printf("WM: debug - inside BoxLoopforall(), submitting to queue\n");                                                \ */
/*       hypre_HandleComputeStream(hypre_handle())->submit([&] (sycl::handler& cgh)                                     \ */
/*          {                                                                                                           \ */
/*             cgh.parallel_for(sycl::nd_range<1>(gridsize*blocksize, blocksize), [=] (sycl::nd_item<1> item)           \ */
/*             { (kernel_name)(item, __VA_ARGS__); } );                                                                 \ */
/*          }).wait_and_throw();                                                                                        \ */
/*    }                                                                                                                 \ */
/* } */



#ifdef __cplusplus
extern "C++" {
#endif

/*********************************************************************
 * forall function
 *********************************************************************/

template<typename LOOP_BODY>
void
BoxLoopforall( LOOP_BODY loop_body,
               HYPRE_Int length )
{
   /* HYPRE_ExecutionPolicy exec_policy = hypre_HandleStructExecPolicy(hypre_handle()); */
   /* WM: TODO: uncomment above and remove below */
   HYPRE_ExecutionPolicy exec_policy = HYPRE_EXEC_DEVICE;

   if (exec_policy == HYPRE_EXEC_HOST)
   {
/* WM: todo - is this really necessary, even? */
/* #ifdef HYPRE_USING_OPENMP */
/* #pragma omp parallel for HYPRE_SMP_SCHEDULE */
/* #endif */
/*       for (HYPRE_Int idx = 0; idx < length; idx++) */
/*       { */
/*          loop_body(idx); */
/*       } */
   }
   else if (exec_policy == HYPRE_EXEC_DEVICE)
   {
      const sycl::range<1> bDim = hypre_GetDefaultCUDABlockDimension();
      const sycl::range<1> gDim = hypre_GetDefaultCUDAGridDimension(length, "thread", bDim);

      hypre_HandleComputeStream(hypre_handle())->submit([&] (sycl::handler& cgh)
         {
            cgh.parallel_for(sycl::nd_range<1>(gDim*bDim, bDim), loop_body);
         }).wait_and_throw();
   }
}

#ifdef __cplusplus
}
#endif

/*********************************************************************
 * Init/Declare/IncK etc.
 *********************************************************************/

/* Get 1-D length of the loop, in hypre__tot */
#define hypre_newBoxLoopInit(ndim, loop_size)              \
   HYPRE_Int hypre__tot = 1;                               \
   for (HYPRE_Int hypre_d = 0; hypre_d < ndim; hypre_d ++) \
   {                                                       \
      hypre__tot *= loop_size[hypre_d];                    \
   }

/* Initialize struct for box-k */
#define hypre_BoxLoopDataDeclareK(k, ndim, loop_size, dbox, start, stride) \
   hypre_Boxloop databox##k;                                               \
   /* dim 0 */                                                             \
   databox##k.lsize0   = loop_size[0];                                     \
   databox##k.strides0 = stride[0];                                        \
   databox##k.bstart0  = start[0] - dbox->imin[0];                         \
   databox##k.bsize0   = dbox->imax[0] - dbox->imin[0];                    \
   /* dim 1 */                                                             \
   if (ndim > 1)                                                           \
   {                                                                       \
      databox##k.lsize1   = loop_size[1];                                  \
      databox##k.strides1 = stride[1];                                     \
      databox##k.bstart1  = start[1] - dbox->imin[1];                      \
      databox##k.bsize1   = dbox->imax[1] - dbox->imin[1];                 \
   }                                                                       \
   else                                                                    \
   {                                                                       \
      databox##k.lsize1   = 1;                                             \
      databox##k.strides1 = 0;                                             \
      databox##k.bstart1  = 0;                                             \
      databox##k.bsize1   = 0;                                             \
   }                                                                       \
   /* dim 2 */                                                             \
   if (ndim == 3)                                                          \
   {                                                                       \
      databox##k.lsize2   = loop_size[2];                                  \
      databox##k.strides2 = stride[2];                                     \
      databox##k.bstart2  = start[2] - dbox->imin[2];                      \
      databox##k.bsize2   = dbox->imax[2] - dbox->imin[2];                 \
   }                                                                       \
   else                                                                    \
   {                                                                       \
      databox##k.lsize2   = 1;                                             \
      databox##k.strides2 = 0;                                             \
      databox##k.bstart2  = 0;                                             \
      databox##k.bsize2   = 0;                                             \
   }

/* Given input 1-D 'idx' in box, get 3-D 'local_idx' in loop_size */
#define hypre_newBoxLoopDeclare(box)                     \
   hypre_Index local_idx;                                \
   size_t idx_local = item.get_local_id(0);                            \
   hypre_IndexD(local_idx, 0)  = idx_local % box.lsize0; \
   idx_local = idx_local / box.lsize0;                   \
   hypre_IndexD(local_idx, 1)  = idx_local % box.lsize1; \
   idx_local = idx_local / box.lsize1;                   \
   hypre_IndexD(local_idx, 2)  = idx_local % box.lsize2; \

/* Given input 3-D 'local_idx', get 1-D 'hypre__i' in 'box' */
#define hypre_BoxLoopIncK(k, box, hypre__i)                                               \
   HYPRE_Int hypre_boxD##k = 1;                                                           \
   HYPRE_Int hypre__i = 0;                                                                \
   hypre__i += (hypre_IndexD(local_idx, 0) * box.strides0 + box.bstart0) * hypre_boxD##k; \
   hypre_boxD##k *= hypre_max(0, box.bsize0 + 1);                                         \
   hypre__i += (hypre_IndexD(local_idx, 1) * box.strides1 + box.bstart1) * hypre_boxD##k; \
   hypre_boxD##k *= hypre_max(0, box.bsize1 + 1);                                         \
   hypre__i += (hypre_IndexD(local_idx, 2) * box.strides2 + box.bstart2) * hypre_boxD##k; \
   hypre_boxD##k *= hypre_max(0, box.bsize2 + 1);



/* BoxLoop 1 */
#define hypre_newBoxLoop1Begin(ndim, loop_size, dbox1, start1, stride1, i1)                           \
{                                                                                                     \
   hypre_newBoxLoopInit(ndim, loop_size);                                                             \
   hypre_BoxLoopDataDeclareK(1, ndim, loop_size, dbox1, start1, stride1);                             \
   BoxLoopforall( [=] (sycl::nd_item<1> item)                                             \
   {                                                                                                  \
      hypre_newBoxLoopDeclare(databox1);                                                              \
      hypre_BoxLoopIncK(1, databox1, i1);

#define hypre_newBoxLoop1End(i1)                                                                      \
   }, hypre__tot);                                                                                                \
}

#define my_hypre_BoxLoop1Begin      hypre_newBoxLoop1Begin
#define my_hypre_BoxLoop1End        hypre_newBoxLoop1End

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

      // WM: question - What's DEVICE_VAR?
#define DEVICE_VAR is_device_ptr(vp)
      my_hypre_BoxLoop1Begin(hypre_StructVectorNDim(vector), loop_size,
                          v_data_box, start, unit_stride, vi);
      {
         vp[vi] = values;
      }
      my_hypre_BoxLoop1End(vi);
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
 * main
 ****************************/

hypre_int
main( hypre_int argc,
      char *argv[] )
{
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

   dim = 1;
   sym  = 1;
   nx = 1000;
   ny = 1;
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
   my_hypre_StructVectorSetConstantValues(x, 1.0);

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
