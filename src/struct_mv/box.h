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

/******************************************************************************
 *
 * Header info for the Box structures
 *
 *****************************************************************************/

#ifndef hypre_BOX_HEADER
#define hypre_BOX_HEADER

#ifndef HYPRE_MAXDIM
#define HYPRE_MAXDIM 3
#endif

/*--------------------------------------------------------------------------
 * hypre_Index:
 *   This is used to define indices in index space, or dimension
 *   sizes of boxes.
 *
 *   The spatial dimensions x, y, and z may be specified by the
 *   integers 0, 1, and 2, respectively (see the hypre_IndexD macro below).
 *   This simplifies the code in the hypre_Box class by reducing code
 *   replication.
 *--------------------------------------------------------------------------*/

typedef HYPRE_Int  hypre_Index[HYPRE_MAXDIM];
typedef HYPRE_Int *hypre_IndexRef;

/*--------------------------------------------------------------------------
 * hypre_Box:
 *--------------------------------------------------------------------------*/

typedef struct hypre_Box_struct
{
   hypre_Index imin;           /* min bounding indices */
   hypre_Index imax;           /* max bounding indices */
   HYPRE_Int   ndim;           /* number of dimensions */
} hypre_Box;

/*--------------------------------------------------------------------------
 * hypre_BoxArray:
 *   An array of boxes.
 *   Since size can be zero, need to store ndim separately.
 *--------------------------------------------------------------------------*/

typedef struct hypre_BoxArray_struct
{
   hypre_Box  *boxes;         /* Array of boxes */
   HYPRE_Int   size;          /* Size of box array */
   HYPRE_Int   alloc_size;    /* Size of currently alloced space */
   HYPRE_Int   ndim;          /* number of dimensions */

} hypre_BoxArray;

#define hypre_BoxArrayExcess 10

/*--------------------------------------------------------------------------
 * hypre_BoxArrayArray:
 *   An array of box arrays.
 *   Since size can be zero, need to store ndim separately.
 *--------------------------------------------------------------------------*/

typedef struct hypre_BoxArrayArray_struct
{
   hypre_BoxArray  **box_arrays;    /* Array of pointers to box arrays */
   HYPRE_Int         size;          /* Size of box array array */
   HYPRE_Int         ndim;          /* number of dimensions */

} hypre_BoxArrayArray;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_Index
 *--------------------------------------------------------------------------*/

#define hypre_IndexD(index, d)  (index[d])

/* Avoid using these macros */
#define hypre_IndexX(index)     hypre_IndexD(index, 0)
#define hypre_IndexY(index)     hypre_IndexD(index, 1)
#define hypre_IndexZ(index)     hypre_IndexD(index, 2)

/*--------------------------------------------------------------------------
 * Member functions: hypre_Index
 *--------------------------------------------------------------------------*/

/*----- Avoid using these Index macros -----*/

#define hypre_SetIndex3(index, ix, iy, iz) \
( hypre_IndexD(index, 0) = ix,\
  hypre_IndexD(index, 1) = iy,\
  hypre_IndexD(index, 2) = iz )

#define hypre_ClearIndex(index)  hypre_SetIndex(index, 0)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_Box
 *--------------------------------------------------------------------------*/

#define hypre_BoxIMin(box)     ((box) -> imin)
#define hypre_BoxIMax(box)     ((box) -> imax)
#define hypre_BoxNDim(box)     ((box) -> ndim)

#define hypre_BoxCopyIndexToData(indexHost,indexData)\
	if(indexData == NULL)\
	{					 \
	    AxCheckError(cudaMalloc((int**)&indexData,sizeof(HYPRE_Int)*(HYPRE_MAXDIM)));	\
	}					 \
	hypre_DataCopyToData(indexHost,indexData,HYPRE_Int,HYPRE_MAXDIM);

#define hypre_BoxSetunitStride(idx) idx = unitstride;
#define hypre_initBoxData(box)					\
	{											\
    	hypre_Index            box_size;\
	    hypre_BoxCopyIndexToData(hypre_BoxIMin(box),hypre_BoxIMinData(box));\
		hypre_BoxCopyIndexToData(hypre_BoxIMax(box),hypre_BoxIMaxData(box)); \
		hypre_BoxGetSize(box, box_size);								\
		hypre_BoxCopyIndexToData(box_size,hypre_BoxSizeData(box));		\
	}																	\
   
	
#define hypre_BoxIMinD(box, d) (hypre_IndexD(hypre_BoxIMin(box), d))
#define hypre_BoxIMaxD(box, d) (hypre_IndexD(hypre_BoxIMax(box), d))
#define hypre_BoxSizeD(box, d) \
hypre_max(0, (hypre_BoxIMaxD(box, d) - hypre_BoxIMinD(box, d) + 1))

#define hypre_IndexDInBox(index, d, box) \
( hypre_IndexD(index, d) >= hypre_BoxIMinD(box, d) && \
  hypre_IndexD(index, d) <= hypre_BoxIMaxD(box, d) )

/* The first hypre_CCBoxIndexRank is better style because it is similar to
   hypre_BoxIndexRank.  The second one sometimes avoids compiler warnings. */
#define hypre_CCBoxIndexRank(box, index) 0
#define hypre_CCBoxIndexRank_noargs() 0
#define hypre_CCBoxOffsetDistance(box, index) 0
  
/*----- Avoid using these Box macros -----*/

#define hypre_BoxSizeX(box)    hypre_BoxSizeD(box, 0)
#define hypre_BoxSizeY(box)    hypre_BoxSizeD(box, 1)
#define hypre_BoxSizeZ(box)    hypre_BoxSizeD(box, 2)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_BoxArray
 *--------------------------------------------------------------------------*/

#define hypre_BoxArrayBoxes(box_array)     ((box_array) -> boxes)
#define hypre_BoxArrayBox(box_array, i)    &((box_array) -> boxes[(i)])
#define hypre_BoxArraySize(box_array)      ((box_array) -> size)
#define hypre_BoxArrayAllocSize(box_array) ((box_array) -> alloc_size)
#define hypre_BoxArrayNDim(box_array)      ((box_array) -> ndim)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_BoxArrayArray
 *--------------------------------------------------------------------------*/

#define hypre_BoxArrayArrayBoxArrays(box_array_array) \
((box_array_array) -> box_arrays)
#define hypre_BoxArrayArrayBoxArray(box_array_array, i) \
((box_array_array) -> box_arrays[(i)])
#define hypre_BoxArrayArraySize(box_array_array) \
((box_array_array) -> size)
#define hypre_BoxArrayArrayNDim(box_array_array) \
((box_array_array) -> ndim)

/*--------------------------------------------------------------------------
 * Looping macros:
 *--------------------------------------------------------------------------*/

#define hypre_ForBoxI(i, box_array) \
for (i = 0; i < hypre_BoxArraySize(box_array); i++)

#define hypre_ForBoxArrayI(i, box_array_array) \
for (i = 0; i < hypre_BoxArrayArraySize(box_array_array); i++)

/*--------------------------------------------------------------------------
 * BoxLoop macros:
 *--------------------------------------------------------------------------*/

#ifdef HYPRE_USE_RAJA
#define hypre_Reductioninit(local_result) \
  HYPRE_Real       local_result;	  \
  local_result = 0.0;

  //ReduceSum< cuda_reduce<BLOCKSIZE>, HYPRE_Real> local_result(0.0);
#else
#define hypre_Reductioninit(local_result) \
HYPRE_Real       local_result;\
local_result = 0.0;
#endif

#if defined(HYPRE_MEMORY_GPU)
static HYPRE_Complex* global_recv_buffer;
static HYPRE_Complex* global_send_buffer;
static HYPRE_Int      global_recv_size = 0;
static HYPRE_Int      global_send_size = 0;

#define hypre_PrepareSendBuffers()				\
  send_buffers_data = hypre_TAlloc(HYPRE_Complex *, num_sends);	\
  if (num_sends > 0)						\
  {								\
     size = hypre_CommPkgSendBufsize(comm_pkg);			\
     if (size > global_send_size)				\
     {							\
         if (global_send_size > 0)						\
	     hypre_DeviceTFree(global_send_buffer);			\
         global_send_buffer = hypre_DeviceCTAlloc(HYPRE_Complex, 5*size); \
         global_send_size   = 5*size;					\
     }									\
     send_buffers_data[0] = global_send_buffer;				\
     for (i = 1; i < num_sends; i++)					\
     {								\
         comm_type = hypre_CommPkgSendType(comm_pkg, i-1);		\
	 size = hypre_CommTypeBufsize(comm_type);			\
	 send_buffers_data[i] = send_buffers_data[i-1] + size;		\
     }									\
  }

#define hypre_PrepareRcevBuffers()		\
  recv_buffers_data = hypre_TAlloc(HYPRE_Complex *, num_recvs);	\
  if (num_recvs > 0)						\
  {								\
      size = hypre_CommPkgRecvBufsize(comm_pkg);		\
      if (size > global_recv_size)				\
      {							\
          if (global_recv_size > 0)				\
	      hypre_DeviceTFree(global_recv_buffer);			\
          global_recv_buffer = hypre_DeviceCTAlloc(HYPRE_Complex, 5*size); \
          global_recv_size   = 5*size;					\
      }									\
      recv_buffers_data[0] = global_recv_buffer;			\
      for (i = 1; i < num_recvs; i++)					\
      {								\
         comm_type = hypre_CommPkgRecvType(comm_pkg, i-1);			\
         size = hypre_CommTypeBufsize(comm_type);			\
         recv_buffers_data[i] = recv_buffers_data[i-1] + size;		\
      }									\
   }

#define hypre_SendBufferTransfer()					\
  if (num_sends > 0)							\
  {									\
      size = hypre_CommPkgSendBufsize(comm_pkg);			\
      dptr = (HYPRE_Complex *) send_buffers[0];				\
      dptr_data = (HYPRE_Complex *) send_buffers_data[0];		\
      hypre_DataCopyFromData(dptr,dptr_data,HYPRE_Complex,size);	\
  }

#define hypre_RecvBufferTransfer()		\
  if (num_recvs > 0)				\
  {						\
      size = 0;					\
      for (i = 0; i < num_recvs; i++)		\
      {						\
          comm_type = hypre_CommPkgRecvType(comm_pkg, i);			\
	  num_entries = hypre_CommTypeNumEntries(comm_type);		\
	  size += hypre_CommTypeBufsize(comm_type);			\
	  if ( hypre_CommPkgFirstComm(comm_pkg) )			\
	  {								\
	      size += hypre_CommPrefixSize(num_entries);		\
	  }								\
      }									\
      dptr = (HYPRE_Complex *) recv_buffers[0];				\
      dptr_data = (HYPRE_Complex *) recv_buffers_data[0];		\
      hypre_DataCopyToData(dptr,dptr_data,HYPRE_Complex,size);		\
   }

#define hypre_FreeBuffer()						\
  hypre_TFree(send_buffers);						\
  hypre_TFree(recv_buffers);						\
  hypre_TFree(send_buffers_data);					\
  hypre_TFree(recv_buffers_data);

#define hypre_MatrixIndexMove(A, stencil_size, i, cdir,size)	\
HYPRE_Int * indices_d;				\
HYPRE_Int indices_h[stencil_size];		\
HYPRE_Int * stencil_shape_d;			\
HYPRE_Int  stencil_shape_h[size*stencil_size];		\
HYPRE_Complex * data_A = hypre_StructMatrixData(A);\
indices_d = hypre_DeviceTAlloc(HYPRE_Int, stencil_size);		\
stencil_shape_d = hypre_DeviceTAlloc(HYPRE_Int, size*stencil_size);	\
for (si = 0; si < stencil_size; si++)				\
  {									\
    HYPRE_Int jj = 0;								\
    indices_h[si]       = hypre_StructMatrixDataIndices(A)[i][si];	\
    if (size > 1) cdir = 0;						\
    stencil_shape_h[si] = hypre_IndexD(stencil_shape[si], cdir); \
    for (jj = 1;jj < size;jj++)						\
      stencil_shape_h[jj*stencil_size+si] = hypre_IndexD(stencil_shape[si], jj);	\
  }									\
hypre_DataCopyToData(indices_h,indices_d,HYPRE_Int,stencil_size);	\
hypre_DataCopyToData(stencil_shape_h,stencil_shape_d,HYPRE_Int,size*stencil_size);

#define hypre_StructGetMatrixBoxData(A, i, si)	(data_A + indices_d[si])

#define hypre_StructGetIndexD(index,i,index_d) (index_d)

#define hypre_StructcleanIndexD() \
  hypre_DeviceTFree(indices_d);			 \
  hypre_DeviceTFree(stencil_shape_d);		 \

#define hypre_StructPreparePrint()		\
  HYPRE_Int tot_size = num_values*hypre_BoxVolume(hypre_BoxArrayBox(data_space, hypre_BoxArraySize(box_array)-1)); \
  data_host = hypre_CTAlloc(HYPRE_Complex, tot_size);			\
  hypre_DataCopyFromData(data_host,data,HYPRE_Complex,tot_size);	\

#define hypre_StructPostPrint() \
  hypre_TFree(data_host);

#else

static HYPRE_Complex* global_recv_buffer;
static HYPRE_Complex* global_send_buffer;
static HYPRE_Int      global_recv_size = 0;
static HYPRE_Int      global_send_size = 0;
typedef struct hypre_Boxloop_struct
{
	HYPRE_Int lsize0,lsize1,lsize2;
	HYPRE_Int strides0,strides1,strides2;
	HYPRE_Int bstart0,bstart1,bstart2;
	HYPRE_Int bsize0,bsize1,bsize2;
} hypre_Boxloop;

#define hypre_PrepareSendBuffers()				\
  send_buffers_data = send_buffers;	\

#define hypre_PrepareRcevBuffers()		\
  recv_buffers_data = recv_buffers;

#define hypre_SendBufferTransfer()  {}

#define hypre_RecvBufferTransfer()  {}


#define hypre_FreeBuffer()						\
  hypre_TFree(send_buffers);						\
  hypre_TFree(recv_buffers);

#define hypre_MatrixIndexMove(A, stencil_size, i, cdir,size) HYPRE_Int* indices_d; HYPRE_Int* stencil_shape_d;
#define hypre_StructGetMatrixBoxData(A, i, si) hypre_StructMatrixBoxData(A,i,si)
#define hypre_StructGetIndexD(index,i,index_d) hypre_IndexD(index,i)
#define hypre_StructcleanIndexD() {;}
#define hypre_StructPreparePrint() data_host = data;
#define hypre_StructPostPrint() {;}

#endif
  
#if 0 /* set to 0 to use the new box loops */

#define HYPRE_BOX_PRIVATE hypre__nx,hypre__ny,hypre__nz,hypre__i,hypre__j,hypre__k

#define hypre_BoxLoopDeclareS(dbox, stride, sx, sy, sz) \
HYPRE_Int  sx = (hypre_IndexX(stride));\
HYPRE_Int  sy = (hypre_IndexY(stride)*hypre_BoxSizeX(dbox));\
HYPRE_Int  sz = (hypre_IndexZ(stride)*\
           hypre_BoxSizeX(dbox)*hypre_BoxSizeY(dbox))

#define hypre_BoxLoopDeclareN(loop_size) \
HYPRE_Int  hypre__i, hypre__j, hypre__k;\
HYPRE_Int  hypre__nx = hypre_IndexX(loop_size);\
HYPRE_Int  hypre__ny = hypre_IndexY(loop_size);\
HYPRE_Int  hypre__nz = hypre_IndexZ(loop_size);\
HYPRE_Int  hypre__mx = hypre__nx;\
HYPRE_Int  hypre__my = hypre__ny;\
HYPRE_Int  hypre__mz = hypre__nz;\
HYPRE_Int  hypre__dir, hypre__max;\
HYPRE_Int  hypre__div, hypre__mod;\
HYPRE_Int  hypre__block, hypre__num_blocks;\
hypre__dir = 0;\
hypre__max = hypre__nx;\
if (hypre__ny > hypre__max)\
{\
   hypre__dir = 1;\
   hypre__max = hypre__ny;\
}\
if (hypre__nz > hypre__max)\
{\
   hypre__dir = 2;\
   hypre__max = hypre__nz;\
}\
hypre__num_blocks = hypre_NumThreads();\
if (hypre__max < hypre__num_blocks)\
{\
   hypre__num_blocks = hypre__max;\
}\
if (hypre__num_blocks > 0)\
{\
   hypre__div = hypre__max / hypre__num_blocks;\
   hypre__mod = hypre__max % hypre__num_blocks;\
}

#define hypre_BoxLoopSet(i, j, k) \
i = 0;\
j = 0;\
k = 0;\
hypre__nx = hypre__mx;\
hypre__ny = hypre__my;\
hypre__nz = hypre__mz;\
if (hypre__num_blocks > 1)\
{\
   if (hypre__dir == 0)\
   {\
      i = hypre__block * hypre__div + hypre_min(hypre__mod, hypre__block);\
      hypre__nx = hypre__div + ((hypre__mod > hypre__block) ? 1 : 0);\
   }\
   else if (hypre__dir == 1)\
   {\
      j = hypre__block * hypre__div + hypre_min(hypre__mod, hypre__block);\
      hypre__ny = hypre__div + ((hypre__mod > hypre__block) ? 1 : 0);\
   }\
   else if (hypre__dir == 2)\
   {\
      k = hypre__block * hypre__div + hypre_min(hypre__mod, hypre__block);\
      hypre__nz = hypre__div + ((hypre__mod > hypre__block) ? 1 : 0);\
   }\
}

#define hypre_BoxLoopGetIndex(index) \
index[0] = hypre__i; index[1] = hypre__j; index[2] = hypre__k

/* Use this before the For macros below to force only one block */
#define hypre_BoxLoopSetOneBlock() hypre__num_blocks = 1

/* Use this to get the block iteration inside a BoxLoop */
#define hypre_BoxLoopBlock() hypre__block

/*-----------------------------------*/

#define hypre_BoxLoop0Begin(ndim, loop_size)\
{\
   hypre_BoxLoopDeclareN(loop_size);

#define hypre_BoxLoop0For()\
   hypre__BoxLoop0For(hypre__i, hypre__j, hypre__k)
#define hypre__BoxLoop0For(i, j, k)\
   for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)\
   {\
   hypre_BoxLoopSet(i, j, k);\
   for (k = 0; k < hypre__nz; k++)\
   {\
      for (j = 0; j < hypre__ny; j++)\
      {\
         for (i = 0; i < hypre__nx; i++)\
         {

#define hypre_BoxLoop0End()\
         }\
      }\
   }\
   }\
}
  
/*-----------------------------------*/

#define hypre_BoxLoop1Begin(ndim, loop_size,\
                            dbox1, start1, stride1, i1)\
{\
   HYPRE_Int  hypre__i1start = hypre_BoxIndexRank(dbox1, start1);\
   hypre_BoxLoopDeclareS(dbox1, stride1, hypre__sx1, hypre__sy1, hypre__sz1);\
   hypre_BoxLoopDeclareN(loop_size);

#define hypre_BoxLoop1For(i1)\
   hypre__BoxLoop1For(hypre__i, hypre__j, hypre__k, i1)
#define hypre__BoxLoop1For(i, j, k, i1)\
   for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)\
   {\
   hypre_BoxLoopSet(i, j, k);\
   i1 = hypre__i1start + i*hypre__sx1 + j*hypre__sy1 + k*hypre__sz1;\
   for (k = 0; k < hypre__nz; k++)\
   {\
      for (j = 0; j < hypre__ny; j++)\
      {\
         for (i = 0; i < hypre__nx; i++)\
         {

#define hypre_BoxLoop1End(i1)\
            i1 += hypre__sx1;\
         }\
         i1 += hypre__sy1 - hypre__nx*hypre__sx1;\
      }\
      i1 += hypre__sz1 - hypre__ny*hypre__sy1;\
   }\
   }\
}
  
/*-----------------------------------*/

#define hypre_BoxLoop2Begin(ndim,loop_size,\
                            dbox1, start1, stride1, i1,\
                            dbox2, start2, stride2, i2)\
{\
   HYPRE_Int  hypre__i1start = hypre_BoxIndexRank(dbox1, start1);\
   HYPRE_Int  hypre__i2start = hypre_BoxIndexRank(dbox2, start2);\
   hypre_BoxLoopDeclareS(dbox1, stride1, hypre__sx1, hypre__sy1, hypre__sz1);\
   hypre_BoxLoopDeclareS(dbox2, stride2, hypre__sx2, hypre__sy2, hypre__sz2);\
   hypre_BoxLoopDeclareN(loop_size);

#define hypre_BoxLoop2For(i1, i2)\
   hypre__BoxLoop2For(hypre__i, hypre__j, hypre__k, i1, i2)
#define hypre__BoxLoop2For(i, j, k, i1, i2)\
   for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)\
   {\
   hypre_BoxLoopSet(i, j, k);\
   i1 = hypre__i1start + i*hypre__sx1 + j*hypre__sy1 + k*hypre__sz1;\
   i2 = hypre__i2start + i*hypre__sx2 + j*hypre__sy2 + k*hypre__sz2;\
   for (k = 0; k < hypre__nz; k++)\
   {\
      for (j = 0; j < hypre__ny; j++)\
      {\
         for (i = 0; i < hypre__nx; i++)\
         {

#define hypre_BoxLoop2End(i1, i2)\
            i1 += hypre__sx1;\
            i2 += hypre__sx2;\
         }\
         i1 += hypre__sy1 - hypre__nx*hypre__sx1;\
         i2 += hypre__sy2 - hypre__nx*hypre__sx2;\
      }\
      i1 += hypre__sz1 - hypre__ny*hypre__sy1;\
      i2 += hypre__sz2 - hypre__ny*hypre__sy2;\
   }\
   }\
}

/*-----------------------------------*/

#define hypre_BoxLoop3Begin(ndim, loop_size,\
                            dbox1, start1, stride1, i1,\
                            dbox2, start2, stride2, i2,\
                            dbox3, start3, stride3, i3)\
{\
   HYPRE_Int  hypre__i1start = hypre_BoxIndexRank(dbox1, start1);\
   HYPRE_Int  hypre__i2start = hypre_BoxIndexRank(dbox2, start2);\
   HYPRE_Int  hypre__i3start = hypre_BoxIndexRank(dbox3, start3);\
   hypre_BoxLoopDeclareS(dbox1, stride1, hypre__sx1, hypre__sy1, hypre__sz1);\
   hypre_BoxLoopDeclareS(dbox2, stride2, hypre__sx2, hypre__sy2, hypre__sz2);\
   hypre_BoxLoopDeclareS(dbox3, stride3, hypre__sx3, hypre__sy3, hypre__sz3);\
   hypre_BoxLoopDeclareN(loop_size);

#define hypre_BoxLoop3For(i1, i2, i3)\
   hypre__BoxLoop3For(hypre__i, hypre__j, hypre__k, i1, i2, i3)
#define hypre__BoxLoop3For(i, j, k, i1, i2, i3)\
   for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)\
   {\
   hypre_BoxLoopSet(i, j, k);\
   i1 = hypre__i1start + i*hypre__sx1 + j*hypre__sy1 + k*hypre__sz1;\
   i2 = hypre__i2start + i*hypre__sx2 + j*hypre__sy2 + k*hypre__sz2;\
   i3 = hypre__i3start + i*hypre__sx3 + j*hypre__sy3 + k*hypre__sz3;\
   for (k = 0; k < hypre__nz; k++)\
   {\
      for (j = 0; j < hypre__ny; j++)\
      {\
         for (i = 0; i < hypre__nx; i++)\
         {

#define hypre_BoxLoop3End(i1, i2, i3)\
            i1 += hypre__sx1;\
            i2 += hypre__sx2;\
            i3 += hypre__sx3;\
         }\
         i1 += hypre__sy1 - hypre__nx*hypre__sx1;\
         i2 += hypre__sy2 - hypre__nx*hypre__sx2;\
         i3 += hypre__sy3 - hypre__nx*hypre__sx3;\
      }\
      i1 += hypre__sz1 - hypre__ny*hypre__sy1;\
      i2 += hypre__sz2 - hypre__ny*hypre__sy2;\
      i3 += hypre__sz3 - hypre__ny*hypre__sy3;\
   }\
   }\
}

/*-----------------------------------*/

#define hypre_BoxLoop4Begin(ndim, loop_size,\
                            dbox1, start1, stride1, i1,\
                            dbox2, start2, stride2, i2,\
                            dbox3, start3, stride3, i3,\
                            dbox4, start4, stride4, i4)\
{\
   HYPRE_Int  hypre__i1start = hypre_BoxIndexRank(dbox1, start1);\
   HYPRE_Int  hypre__i2start = hypre_BoxIndexRank(dbox2, start2);\
   HYPRE_Int  hypre__i3start = hypre_BoxIndexRank(dbox3, start3);\
   HYPRE_Int  hypre__i4start = hypre_BoxIndexRank(dbox4, start4);\
   hypre_BoxLoopDeclareS(dbox1, stride1, hypre__sx1, hypre__sy1, hypre__sz1);\
   hypre_BoxLoopDeclareS(dbox2, stride2, hypre__sx2, hypre__sy2, hypre__sz2);\
   hypre_BoxLoopDeclareS(dbox3, stride3, hypre__sx3, hypre__sy3, hypre__sz3);\
   hypre_BoxLoopDeclareS(dbox4, stride4, hypre__sx4, hypre__sy4, hypre__sz4);\
   hypre_BoxLoopDeclareN(loop_size);

#define hypre_BoxLoop4For(i1, i2, i3, i4)\
   hypre__BoxLoop4For(hypre__i, hypre__j, hypre__k, i1, i2, i3, i4)
#define hypre__BoxLoop4For(i, j, k, i1, i2, i3, i4)\
   for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)\
   {\
   hypre_BoxLoopSet(i, j, k);\
   i1 = hypre__i1start + i*hypre__sx1 + j*hypre__sy1 + k*hypre__sz1;\
   i2 = hypre__i2start + i*hypre__sx2 + j*hypre__sy2 + k*hypre__sz2;\
   i3 = hypre__i3start + i*hypre__sx3 + j*hypre__sy3 + k*hypre__sz3;\
   i4 = hypre__i4start + i*hypre__sx4 + j*hypre__sy4 + k*hypre__sz4;\
   for (k = 0; k < hypre__nz; k++)\
   {\
      for (j = 0; j < hypre__ny; j++)\
      {\
         for (i = 0; i < hypre__nx; i++)\
         {

#define hypre_BoxLoop4End(i1, i2, i3, i4)\
            i1 += hypre__sx1;\
            i2 += hypre__sx2;\
            i3 += hypre__sx3;\
            i4 += hypre__sx4;\
         }\
         i1 += hypre__sy1 - hypre__nx*hypre__sx1;\
         i2 += hypre__sy2 - hypre__nx*hypre__sx2;\
         i3 += hypre__sy3 - hypre__nx*hypre__sx3;\
         i4 += hypre__sy4 - hypre__nx*hypre__sx4;\
      }\
      i1 += hypre__sz1 - hypre__ny*hypre__sy1;\
      i2 += hypre__sz2 - hypre__ny*hypre__sy2;\
      i3 += hypre__sz3 - hypre__ny*hypre__sy3;\
      i4 += hypre__sz4 - hypre__ny*hypre__sy4;\
   }\
   }\
}

/*-----------------------------------*/

#else

#define hypre_SerialBoxLoop0Begin(ndim, loop_size)\
{\
   zypre_BoxLoopDeclare();\
   zypre_BoxLoopInit(ndim, loop_size);					\
   hypre_BoxLoopSetOneBlock();						\
   for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)\
   {\
      zypre_BoxLoopSet();\
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)\
      {\
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)\
         {

#define hypre_SerialBoxLoop0End()\
         }\
         zypre_BoxLoopInc1();\
         zypre_BoxLoopInc2();\
      }\
   }\
}
#define hypre_SerialBoxLoop1Begin(ndim, loop_size,				\
				  dbox1, start1, stride1, i1)		\
{									\
    HYPRE_Int i1;								\
    zypre_BoxLoopDeclare();						\
    zypre_BoxLoopDeclareK(1);						\
    zypre_BoxLoopInit(ndim, loop_size);					\
    zypre_BoxLoopInitK(1, dbox1, start1, stride1, i1);			\
    zypre_BoxLoopSetOneBlock();						\
    for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++) \
    {									\
        zypre_BoxLoopSet();							\
	zypre_BoxLoopSetK(1, i1);					\
	for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)		\
	{								\
	  for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)		\
	  {
  
#define hypre_SerialBoxLoop1End(i1)				\
             i1 += hypre__i0inc1;							\
          }									\
	  zypre_BoxLoopInc1();					\
	  i1 += hypre__ikinc1[hypre__d];				\
	  zypre_BoxLoopInc2();						\
       }									\
   }									\
}

#define hypre_SerialBoxLoop2Begin(ndim, loop_size,\
							   dbox1, start1, stride1, i1,	\
							   dbox2, start2, stride2, i2)	\
{\
   zypre_BoxLoopDeclare();\
   zypre_BoxLoopDeclareK(1);\
   zypre_BoxLoopDeclareK(2);\
   zypre_BoxLoopInit(ndim, loop_size);\
   zypre_BoxLoopInitK(1, dbox1, start1, stride1, i1);\
   zypre_BoxLoopInitK(2, dbox2, start2, stride2, i2);\
   zypre_BoxLoopSetOneBlock();						\
   for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)	\
   {\
      zypre_BoxLoopSet();\
      zypre_BoxLoopSetK(1, i1);\
      zypre_BoxLoopSetK(2, i2);\
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)\
      {\
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)\
         {

#define hypre_SerialBoxLoop2End(i1, i2)\
            i1 += hypre__i0inc1;\
            i2 += hypre__i0inc2;\
         }\
         zypre_BoxLoopInc1();\
         i1 += hypre__ikinc1[hypre__d];\
         i2 += hypre__ikinc2[hypre__d];\
         zypre_BoxLoopInc2();\
      }\
   }\
}
  
#define HYPRE_BOX_PRIVATE        ZYPRE_BOX_PRIVATE

#endif /* end if 1 */

#endif

/******************************************************************************
 *
 * NEW BoxLoop STUFF
 *
 *****************************************************************************/

#ifndef hypre_ZBOX_HEADER
#define hypre_ZBOX_HEADER

#define ZYPRE_BOX_PRIVATE hypre__IN,hypre__JN,hypre__I,hypre__J,hypre__d,hypre__i

/*--------------------------------------------------------------------------
 * BoxLoop macros:
 *--------------------------------------------------------------------------*/

#define zypre_BoxLoopDeclare() \
HYPRE_Int  hypre__tot, hypre__div, hypre__mod;\
HYPRE_Int  hypre__block, hypre__num_blocks;\
HYPRE_Int  hypre__d, hypre__ndim;\
HYPRE_Int  hypre__I, hypre__J, hypre__IN, hypre__JN;\
HYPRE_Int  hypre__i[HYPRE_MAXDIM+1], hypre__n[HYPRE_MAXDIM+1]

#define zypre_BoxLoopDeclareK(k) \
HYPRE_Int  hypre__ikstart##k, hypre__i0inc##k;\
HYPRE_Int  hypre__sk##k[HYPRE_MAXDIM], hypre__ikinc##k[HYPRE_MAXDIM+1]

#define zypre_BoxLoopInit(ndim, loop_size) \
hypre__ndim = ndim;\
hypre__n[0] = loop_size[0];\
hypre__tot = 1;\
for (hypre__d = 1; hypre__d < hypre__ndim; hypre__d++)\
{\
   hypre__n[hypre__d] = loop_size[hypre__d];\
   hypre__tot *= hypre__n[hypre__d];\
}\
hypre__n[hypre__ndim] = 2;\
hypre__num_blocks = hypre_NumThreads();\
if (hypre__tot < hypre__num_blocks)\
{\
   hypre__num_blocks = hypre__tot;\
}\
if (hypre__num_blocks > 0)\
{\
   hypre__div = hypre__tot / hypre__num_blocks;\
   hypre__mod = hypre__tot % hypre__num_blocks;\
}

#define zypre_BoxLoopInitK(k, dboxk, startk, stridek, ik) \
hypre__sk##k[0] = stridek[0];\
hypre__ikinc##k[0] = 0;\
ik = hypre_BoxSizeD(dboxk, 0); /* temporarily use ik */\
for (hypre__d = 1; hypre__d < hypre__ndim; hypre__d++)\
{\
   hypre__sk##k[hypre__d] = ik*stridek[hypre__d];\
   hypre__ikinc##k[hypre__d] = hypre__ikinc##k[hypre__d-1] +\
      hypre__sk##k[hypre__d] - hypre__n[hypre__d-1]*hypre__sk##k[hypre__d-1];\
   ik *= hypre_BoxSizeD(dboxk, hypre__d);\
}\
hypre__i0inc##k = hypre__sk##k[0];\
hypre__ikinc##k[hypre__ndim] = 0;\
hypre__ikstart##k = hypre_BoxIndexRank(dboxk, startk)

#define zypre_BoxLoopSet() \
hypre__IN = hypre__n[0];\
if (hypre__num_blocks > 1)/* in case user sets num_blocks to 1 */\
{\
   hypre__JN = hypre__div + ((hypre__mod > hypre__block) ? 1 : 0);\
   hypre__J = hypre__block * hypre__div + hypre_min(hypre__mod, hypre__block);\
   for (hypre__d = 1; hypre__d < hypre__ndim; hypre__d++)\
   {\
      hypre__i[hypre__d] = hypre__J % hypre__n[hypre__d];\
      hypre__J /= hypre__n[hypre__d];\
   }\
}\
else\
{\
   hypre__JN = hypre__tot;\
   for (hypre__d = 1; hypre__d < hypre__ndim; hypre__d++)\
   {\
      hypre__i[hypre__d] = 0;\
   }\
}\
hypre__i[hypre__ndim] = 0

#define zypre_BoxLoopSetK(k, ik) \
ik = hypre__ikstart##k;\
for (hypre__d = 1; hypre__d < hypre__ndim; hypre__d++)\
{\
   ik += hypre__i[hypre__d]*hypre__sk##k[hypre__d];\
}

#define zypre_BoxLoopInc1() \
hypre__d = 1;\
while ((hypre__i[hypre__d]+2) > hypre__n[hypre__d])\
{\
   hypre__d++;\
}

#define zypre_BoxLoopInc2() \
hypre__i[hypre__d]++;\
while (hypre__d > 1)\
{\
   hypre__d--;\
   hypre__i[hypre__d] = 0;\
}

/* This returns the loop index (of type hypre_Index) for the current iteration,
 * where the numbering starts at 0.  It works even when threading is turned on,
 * as long as 'index' is declared to be private. */
#define zypre_BoxLoopGetIndex(index) \
index[0] = hypre__I;\
for (hypre__d = 1; hypre__d < hypre__ndim; hypre__d++)\
{\
   index[hypre__d] = hypre__i[hypre__d];\
}

/* Use this before the For macros below to force only one block */
#define zypre_BoxLoopSetOneBlock() hypre__num_blocks = 1

/* Use this to get the block iteration inside a BoxLoop */
#define zypre_BoxLoopBlock() hypre__block

/*-----------------------------------*/

#define zypre_BoxLoop0Begin(ndim, loop_size)\
{\
   zypre_BoxLoopDeclare();\
   zypre_BoxLoopInit(ndim, loop_size);

#define zypre_BoxLoop0For()\
   for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)\
   {\
      zypre_BoxLoopSet();\
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)\
      {\
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)\
         {

#define zypre_BoxLoop0End()\
         }\
         zypre_BoxLoopInc1();\
         zypre_BoxLoopInc2();\
      }\
   }\
}

/*-----------------------------------*/

#define zypre_BoxLoop1Begin(ndim, loop_size,\
                            dbox1, start1, stride1, i1)\
{\
   HYPRE_Int i1;					\
   zypre_BoxLoopDeclare();\
   zypre_BoxLoopDeclareK(1);\
   zypre_BoxLoopInit(ndim, loop_size);\
   zypre_BoxLoopInitK(1, dbox1, start1, stride1, i1);

#define zypre_BoxLoop1For(i1)\
   for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)\
   {\
      HYPRE_Int i1;					\
      zypre_BoxLoopSet();\
      zypre_BoxLoopSetK(1, i1);\
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)\
      {\
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)\
         {

#define zypre_BoxLoop1End(i1)\
            i1 += hypre__i0inc1;\
         }\
         zypre_BoxLoopInc1();\
         i1 += hypre__ikinc1[hypre__d];\
         zypre_BoxLoopInc2();\
      }\
   }\
}

/*-----------------------------------*/

#define zypre_BoxLoop2Begin(ndim, loop_size,\
                            dbox1, start1, stride1, i1,\
                            dbox2, start2, stride2, i2)\
{\
   HYPRE_Int i1,i2;				\
   zypre_BoxLoopDeclare();\
   zypre_BoxLoopDeclareK(1);\
   zypre_BoxLoopDeclareK(2);\
   zypre_BoxLoopInit(ndim, loop_size);\
   zypre_BoxLoopInitK(1, dbox1, start1, stride1, i1);\
   zypre_BoxLoopInitK(2, dbox2, start2, stride2, i2);

#define zypre_BoxLoop2For(i1, i2)\
   for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)\
   {\
      HYPRE_Int i1,i2;				\
      zypre_BoxLoopSet();\
      zypre_BoxLoopSetK(1, i1);\
      zypre_BoxLoopSetK(2, i2);\
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)\
      {\
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)\
         {

#define zypre_BoxLoop2End(i1, i2)\
            i1 += hypre__i0inc1;\
            i2 += hypre__i0inc2;\
         }\
         zypre_BoxLoopInc1();\
         i1 += hypre__ikinc1[hypre__d];\
         i2 += hypre__ikinc2[hypre__d];\
         zypre_BoxLoopInc2();\
      }\
   }\
}

/*-----------------------------------*/

#define zypre_BoxLoop3Begin(ndim, loop_size,\
                            dbox1, start1, stride1, i1,\
                            dbox2, start2, stride2, i2,\
                            dbox3, start3, stride3, i3)\
{\
   HYPRE_Int i1,i2,i3;				\
   zypre_BoxLoopDeclare();\
   zypre_BoxLoopDeclareK(1);\
   zypre_BoxLoopDeclareK(2);\
   zypre_BoxLoopDeclareK(3);\
   zypre_BoxLoopInit(ndim, loop_size);\
   zypre_BoxLoopInitK(1, dbox1, start1, stride1, i1);\
   zypre_BoxLoopInitK(2, dbox2, start2, stride2, i2);\
   zypre_BoxLoopInitK(3, dbox3, start3, stride3, i3);

#define zypre_BoxLoop3For(i1, i2, i3)\
   for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)\
   {\
      HYPRE_Int i1,i2,i3;				\
      zypre_BoxLoopSet();\
      zypre_BoxLoopSetK(1, i1);\
      zypre_BoxLoopSetK(2, i2);\
      zypre_BoxLoopSetK(3, i3);\
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)\
      {\
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)\
         {

#define zypre_BoxLoop3End(i1, i2, i3)\
            i1 += hypre__i0inc1;\
            i2 += hypre__i0inc2;\
            i3 += hypre__i0inc3;\
         }\
         zypre_BoxLoopInc1();\
         i1 += hypre__ikinc1[hypre__d];\
         i2 += hypre__ikinc2[hypre__d];\
         i3 += hypre__ikinc3[hypre__d];\
         zypre_BoxLoopInc2();\
      }\
   }\
}

/*-----------------------------------*/

#define zypre_BoxLoop4Begin(ndim, loop_size,\
                            dbox1, start1, stride1, i1,\
                            dbox2, start2, stride2, i2,\
                            dbox3, start3, stride3, i3,\
                            dbox4, start4, stride4, i4)\
{\
   HYPRE_Int i1,i2,i3,i4;				\
   zypre_BoxLoopDeclare();\
   zypre_BoxLoopDeclareK(1);\
   zypre_BoxLoopDeclareK(2);\
   zypre_BoxLoopDeclareK(3);\
   zypre_BoxLoopDeclareK(4);\
   zypre_BoxLoopInit(ndim, loop_size);\
   zypre_BoxLoopInitK(1, dbox1, start1, stride1, i1);\
   zypre_BoxLoopInitK(2, dbox2, start2, stride2, i2);\
   zypre_BoxLoopInitK(3, dbox3, start3, stride3, i3);\
   zypre_BoxLoopInitK(4, dbox4, start4, stride4, i4);

#define zypre_BoxLoop4For(i1, i2, i3, i4)\
   for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)\
   {\
      HYPRE_Int i1,i2,i3,i4;				\
      zypre_BoxLoopSet();\
      zypre_BoxLoopSetK(1, i1);\
      zypre_BoxLoopSetK(2, i2);\
      zypre_BoxLoopSetK(3, i3);\
      zypre_BoxLoopSetK(4, i4);\
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)\
      {\
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)\
         {

#define zypre_BoxLoop4End(i1, i2, i3, i4)\
            i1 += hypre__i0inc1;\
            i2 += hypre__i0inc2;\
            i3 += hypre__i0inc3;\
            i4 += hypre__i0inc4;\
         }\
         zypre_BoxLoopInc1();\
         i1 += hypre__ikinc1[hypre__d];\
         i2 += hypre__ikinc2[hypre__d];\
         i3 += hypre__ikinc3[hypre__d];\
         i4 += hypre__ikinc4[hypre__d];\
         zypre_BoxLoopInc2();\
      }\
   }\
}

/*-----------------------------------*/

#endif




/*--------------------------------------------------------------------------
 * NOTES - Keep these for reference here and elsewhere in the code
 *--------------------------------------------------------------------------*/

#if 0

#define hypre_BoxLoop2Begin(loop_size,
                            dbox1, start1, stride1, i1,
                            dbox2, start2, stride2, i2)
{
   /* init hypre__i1start */
   HYPRE_Int  hypre__i1start = hypre_BoxIndexRank(dbox1, start1);
   HYPRE_Int  hypre__i2start = hypre_BoxIndexRank(dbox2, start2);
   /* declare and set hypre__s1 */
   hypre_BoxLoopDeclareS(dbox1, stride1, hypre__sx1, hypre__sy1, hypre__sz1);
   hypre_BoxLoopDeclareS(dbox2, stride2, hypre__sx2, hypre__sy2, hypre__sz2);
   /* declare and set hypre__n, hypre__m, hypre__dir, hypre__max,
    *                 hypre__div, hypre__mod, hypre__block, hypre__num_blocks */
   hypre_BoxLoopDeclareN(loop_size);

#define hypre_BoxLoop2For(i, j, k, i1, i2)
   for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)
   {
   /* set i and hypre__n */
   hypre_BoxLoopSet(i, j, k);
   /* set i1 */
   i1 = hypre__i1start + i*hypre__sx1 + j*hypre__sy1 + k*hypre__sz1;
   i2 = hypre__i2start + i*hypre__sx2 + j*hypre__sy2 + k*hypre__sz2;
   for (k = 0; k < hypre__nz; k++)
   {
      for (j = 0; j < hypre__ny; j++)
      {
         for (i = 0; i < hypre__nx; i++)
         {

#define hypre_BoxLoop2End(i1, i2)
            i1 += hypre__sx1;
            i2 += hypre__sx2;
         }
         i1 += hypre__sy1 - hypre__nx*hypre__sx1;
         i2 += hypre__sy2 - hypre__nx*hypre__sx2;
      }
      i1 += hypre__sz1 - hypre__ny*hypre__sy1;
      i2 += hypre__sz2 - hypre__ny*hypre__sy2;
   }
   }
}

/*----------------------------------------
 * Idea 2: Simple version of Idea 3 below
 *----------------------------------------*/

N = 1;
for (d = 0; d < ndim; d++)
{
   N *= n[d];
   i[d] = 0;
   n[d] -= 2; /* this produces a simpler comparison below */
}
i[ndim] = 0;
n[ndim] = 0;
for (I = 0; I < N; I++)
{
   /* loop body */

   for (d = 0; i[d] > n[d]; d++)
   {
      i[d] = 0;
   }
   i[d]++;
   i1 += s1[d]; /* NOTE: These are different from hypre__sx1, etc. above */
   i2 += s2[d]; /* The lengths of i, n, and s must be (ndim+1) */
}

/*----------------------------------------
 * Idea 3: Approach used in the box loops
 *----------------------------------------*/

N = 1;
for (d = 1; d < ndim; d++)
{
   N *= n[d];
   i[d] = 0;
   n[d] -= 2; /* this produces a simpler comparison below */
}
i[ndim] = 0;
n[ndim] = 0;
for (J = 0; J < N; J++)
{
   for (I = 0; I < n[0]; I++)
   {
      /* loop body */

      i1 += s1[0];
      i2 += s2[0];
   }
   for (d = 1; i[d] > n[d]; d++)
   {
      i[d] = 0;
   }
   i[d]++;
   i1 += s1[d]; /* NOTE: These are different from hypre__sx1, etc. above */
   i2 += s2[d]; /* The lengths of i, n, and s must be (ndim+1) */
}

#endif
