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
 * Header info for the BoxLoop
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * BoxLoop macros:
 *--------------------------------------------------------------------------*/

#ifndef HYPRE_NEWBOXLOOP_HEADER
#define HYPRE_NEWBOXLOOP_HEADER
#ifdef HYPRE_USING_OPENMP
#define Pragma(x) _Pragma(#x)
#define OMP1(args...) Pragma(omp parallel for private(ZYPRE_BOX_PRIVATE,args) HYPRE_SMP_SCHEDULE)
#define OMPREDUCTION() Pragma(omp parallel for private(HYPRE_BOX_PRIVATE,HYPRE_BOX_PRIVATE_VAR) HYPRE_BOX_REDUCTION HYPRE_SMP_SCHEDULE)
#else
#define OMP1(args...)  ;
#define OMPREDUCTION() ;
#endif

#define hypre_rand(val) \
{\
    val = rand();\
}

#define zypre_newBoxLoop0Begin(ndim, loop_size)				\
{\
   zypre_BoxLoopDeclare();									\
   zypre_BoxLoopInit(ndim, loop_size);						\
   OMP1(HYPRE_BOX_PRIVATE_VAR)								\
   for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)\
   {\
      zypre_BoxLoopSet();\
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)\
      {\
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)\
         {

#define zypre_newBoxLoop0End()\
         }\
         zypre_BoxLoopInc1();\
         zypre_BoxLoopInc2();\
      }\
   }\
}

#define zypre_newBoxLoop1Begin(ndim, loop_size,				\
							   dbox1, start1, stride1, i1) 	\
	{														\
	zypre_BoxLoopDeclare();									\
	zypre_BoxLoopDeclareK(1);								\
	zypre_BoxLoopInit(ndim, loop_size);						\
	zypre_BoxLoopInitK(1, dbox1, start1, stride1, i1);					\
	OMP1(HYPRE_BOX_PRIVATE_VAR)															\
	for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++) \
	{																	\
		zypre_BoxLoopSet();												\
		zypre_BoxLoopSetK(1, i1);										\
		for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)			\
		{																\
			for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)		\
			{

#define zypre_newBoxLoop1End(i1)				\
	             i1 += hypre__i0inc1;						\
		    }											\
			zypre_BoxLoopInc1();					\
	        i1 += hypre__ikinc1[hypre__d];				\
	        zypre_BoxLoopInc2();						\
		}											\
	}											\
}


#define zypre_newBoxLoop2Begin(ndim, loop_size,\
							   dbox1, start1, stride1, i1,	\
							   dbox2, start2, stride2, i2)	\
{\
   zypre_BoxLoopDeclare();\
   zypre_BoxLoopDeclareK(1);\
   zypre_BoxLoopDeclareK(2);\
   zypre_BoxLoopInit(ndim, loop_size);\
   zypre_BoxLoopInitK(1, dbox1, start1, stride1, i1);\
   zypre_BoxLoopInitK(2, dbox2, start2, stride2, i2);\
   OMP1(HYPRE_BOX_PRIVATE_VAR)															\
   for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)	\
   {\
      zypre_BoxLoopSet();\
      zypre_BoxLoopSetK(1, i1);\
      zypre_BoxLoopSetK(2, i2);\
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)\
      {\
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)\
         {

#define zypre_newBoxLoop2End(i1, i2)\
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


#define zypre_newBoxLoop3Begin(ndim, loop_size,\
							   dbox1, start1, stride1, i1,	\
							   dbox2, start2, stride2, i2,	\
							   dbox3, start3, stride3, i3)	\
{														\
   zypre_BoxLoopDeclare();									\
   zypre_BoxLoopDeclareK(1);								\
   zypre_BoxLoopDeclareK(2);								\
   zypre_BoxLoopDeclareK(3);								\
   zypre_BoxLoopInit(ndim, loop_size);						\
   zypre_BoxLoopInitK(1, dbox1, start1, stride1, i1);		\
   zypre_BoxLoopInitK(2, dbox2, start2, stride2, i2);		\
   zypre_BoxLoopInitK(3, dbox3, start3, stride3, i3);		\
   OMP1(HYPRE_BOX_PRIVATE_VAR)								\
   for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)	\
   {\
      zypre_BoxLoopSet();\
      zypre_BoxLoopSetK(1, i1);\
      zypre_BoxLoopSetK(2, i2);\
      zypre_BoxLoopSetK(3, i3);\
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)\
      {\
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)\
         {

#define zypre_newBoxLoop3End(i1, i2, i3)\
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

#define zypre_newBoxLoop4Begin(ndim, loop_size,\
                            dbox1, start1, stride1, i1,\
                            dbox2, start2, stride2, i2,\
                            dbox3, start3, stride3, i3,\
                            dbox4, start4, stride4, i4)\
{\
   zypre_BoxLoopDeclare();\
   zypre_BoxLoopDeclareK(1);\
   zypre_BoxLoopDeclareK(2);\
   zypre_BoxLoopDeclareK(3);\
   zypre_BoxLoopDeclareK(4);\
   zypre_BoxLoopInit(ndim, loop_size);\
   zypre_BoxLoopInitK(1, dbox1, start1, stride1, i1);\
   zypre_BoxLoopInitK(2, dbox2, start2, stride2, i2);\
   zypre_BoxLoopInitK(3, dbox3, start3, stride3, i3);\
   zypre_BoxLoopInitK(4, dbox4, start4, stride4, i4);\
   OMP1(HYPRE_BOX_PRIVATE_VAR)								\
   for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)\
   {\
      zypre_BoxLoopSet();\
      zypre_BoxLoopSetK(1, i1);\
      zypre_BoxLoopSetK(2, i2);\
      zypre_BoxLoopSetK(3, i3);\
      zypre_BoxLoopSetK(4, i4);\
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)\
      {\
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)\
         {

#define zypre_newBoxLoop4End(i1, i2, i3, i4)\
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

#define zypre_newBoxLoop1ReductionBegin(ndim, loop_size,		\
					dbox1, start1, stride1, i1,	\
                                        sum)				\
{									\
   zypre_BoxLoopDeclare();						\
   zypre_BoxLoopDeclareK(1);						\
   zypre_BoxLoopInit(ndim, loop_size);					\
   zypre_BoxLoopInitK(1, dbox1, start1, stride1, i1);			\
   OMPREDUCTION()							\
   for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++) \
   {\
      zypre_BoxLoopSet();\
      zypre_BoxLoopSetK(1, i1);\
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)\
      {\
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)\
         {

#define zypre_newBoxLoop1ReductionEnd(i1, sum)\
            i1 += hypre__i0inc1;\
         }\
         zypre_BoxLoopInc1();\
         i1 += hypre__ikinc1[hypre__d];\
         zypre_BoxLoopInc2();\
      }\
   }\
}

#define hypre_newBoxLoop2ReductionBegin(ndim, loop_size,				\
					dbox1, start1, stride1, i1,	\
					dbox2, start2, stride2, i2,	\
                                        sum)							\
{\
   HYPRE_Int i1,i2;				\
   zypre_BoxLoopDeclare();\
   zypre_BoxLoopDeclareK(1);\
   zypre_BoxLoopDeclareK(2);\
   zypre_BoxLoopInit(ndim, loop_size);\
   zypre_BoxLoopInitK(1, dbox1, start1, stride1, i1);\
   zypre_BoxLoopInitK(2, dbox2, start2, stride2, i2);\
   OMPREDUCTION()														\
   for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)	\
   {\
      zypre_BoxLoopSet();\
      zypre_BoxLoopSetK(1, i1);\
      zypre_BoxLoopSetK(2, i2);\
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)\
      {\
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)\
         {

#define hypre_newBoxLoop2ReductionEnd(i1, i2, sum)\
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

#define hypre_LoopBegin(size,idx)			\
{									\
   HYPRE_Int idx;							\
   for (idx = 0;idx < size;idx ++)					\
   {

#define hypre_LoopEnd()					\
  }							\
}

#define hypre_BoxLoopSetOneBlock zypre_BoxLoopSetOneBlock


#define hypre_BoxLoop0Begin      zypre_BoxLoop0Begin
#define hypre_BoxLoop0For        zypre_BoxLoop0For
#define hypre_BoxLoop0End        zypre_BoxLoop0End
#define hypre_BoxLoop1Begin      zypre_BoxLoop1Begin
#define hypre_BoxLoop1For        zypre_BoxLoop1For
#define hypre_BoxLoop1End        zypre_BoxLoop1End
#define hypre_BoxLoop2Begin      zypre_BoxLoop2Begin
#define hypre_BoxLoop2For        zypre_BoxLoop2For
#define hypre_BoxLoop2End        zypre_BoxLoop2End
#define hypre_BoxLoop3Begin      zypre_BoxLoop3Begin
#define hypre_BoxLoop3For        zypre_BoxLoop3For
#define hypre_BoxLoop3End        zypre_BoxLoop3End
#define hypre_BoxLoop4Begin      zypre_BoxLoop4Begin
#define hypre_BoxLoop4For        zypre_BoxLoop4For
#define hypre_BoxLoop4End        zypre_BoxLoop4End
