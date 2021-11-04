/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header file for multiprecision utilities
 *
 *****************************************************************************/
 
#ifndef MULTIPRECISION_UTILITIES_HEADER
#define MULTIPRECISION_UTILITIES_HEADER

#define CONCAT2_(a, b) a ## _ ## b
#define CONCAT_(a, b) CONCAT2_(a, b)

/* multiprecision build types */
#define FLT_SUFFIX flt
#define DBL_SUFFIX dbl
#define LDBL_SUFFIX long_dbl

/*--------------------------------------------------------------------------
* For Multi-precision build. Only set when hypre 
* is built with mixed-precision
*---------------------------------------------------------------------------*/
#if defined(HYPRE_MIXED_PRECISION)
/* matrix/ solver precision options */
typedef enum 
{
   HYPRE_REAL_SINGLE,
   HYPRE_REAL_DOUBLE,
   HYPRE_REAL_LONGDOUBLE
} HYPRE_Precision;
/*--------------------------------------------------------------------------
* Reset build types for Multi-precision build
*---------------------------------------------------------------------------*/
#if defined(MP_BUILD_SINGLE)
#define HYPRE_OBJECT_PRECISION HYPRE_REAL_SINGLE
#define BUILD_MP_FUNC 1
#undef HYPRE_LONG_DOUBLE
#ifndef HYPRE_SINGLE
#define HYPRE_SINGLE 1
#endif
#elif defined(MP_BUILD_LONGDOUBLE)
#define HYPRE_OBJECT_PRECISION HYPRE_REAL_LONGDOUBLE
#define BUILD_MP_FUNC 1
#undef HYPRE_SINGLE
#ifndef HYPRE_LONG_DOUBLE
#define HYPRE_LONG_DOUBLE 1
#endif
#elif defined(MP_BUILD_DOUBLE)
#define HYPRE_OBJECT_PRECISION HYPRE_REAL_DOUBLE
#define BUILD_MP_FUNC 1
#undef HYPRE_SINGLE
#undef HYPRE_LONG_DOUBLE
#else
#ifdef BUILD_MP_FUNC
#undef BUILD_MP_FUNC
#endif
#define UNDEFINED_MP_BUILD 1
#endif
/*--------------------------------------------------------------------------
 * HYPRE multiprecision extensions
 *--------------------------------------------------------------------------*/
/* Macro to generate typed functions */
#if defined(HYPRE_SINGLE)
//#define FUNC_SUFFIX flt
#define HYPRE_TYPED_FUNC(a) CONCAT_(a, FLT_SUFFIX)
#elif defined(HYPRE_LONG_DOUBLE)
//#define FUNC_SUFFIX long_dbl
#define HYPRE_TYPED_FUNC(a) CONCAT_(a, LDBL_SUFFIX)
#else /* HYPRE_DOUBLE */
//#define FUNC_SUFFIX dbl
#define HYPRE_TYPED_FUNC(a) CONCAT_(a, DBL_SUFFIX)
#endif

/* Apply suffix to define typed function */
//#define HYPRE_TYPED_FUNC(a) CONCAT_(a, FUNC_SUFFIX)

#else
/* define no-op for typed function macro */
#define HYPRE_TYPED_FUNC(a) a
#define BUILD_MP_FUNC 1
#endif

/* Helper macros to generate multiprecision function declarations */
#define DECLARE_MP_FUNC(rtype,func,fargs...)\
	rtype CONCAT_(func,FLT_SUFFIX) (fargs);\
	rtype CONCAT_(func,DBL_SUFFIX) (fargs);\
	rtype CONCAT_(func,LDBL_SUFFIX) (fargs);\

#define DECLARE_DP_FUNC(rtype,func,fargs...)\
	rtype CONCAT_(func,DBL_SUFFIX) (fargs);\

#define HYPRE_DP_FUNC(a) CONCAT_(a, DBL_SUFFIX)

#define DECLARE_SP_FUNC(rtype,func,fargs...)\
	rtype CONCAT_(func,FLT_SUFFIX) (fargs);\
	
#define HYPRE_SP_FUNC(a) CONCAT_(a, FLT_SUFFIX)	

/* code for scalar or void return type */
#define MP_METHOD_FUNC(precision,func,args...)\
	switch(precision) {\
	   case HYPRE_REAL_SINGLE: \
	      return CONCAT_(func,FLT_SUFFIX) (args);\
	   case HYPRE_REAL_DOUBLE: \
	      return CONCAT_(func,DBL_SUFFIX) (args);\
	   case HYPRE_REAL_LONGDOUBLE: \
	      return CONCAT_(func,LDBL_SUFFIX) (args);\
	   default:\
	      hypre_printf("Unknown solver precision" );\
	      exit(0);\
        }\

/* code for pointer return type */        
#define MP_METHOD_FUNCPTR(rval,precision,func,args...)\
	switch(precision) {\
	   case HYPRE_REAL_SINGLE: \
	      rval = CONCAT_(func,FLT_SUFFIX) (args);\
	   case HYPRE_REAL_DOUBLE: \
	      rval = CONCAT_(func,DBL_SUFFIX) (args);\
	   case HYPRE_REAL_LONGDOUBLE: \
	      rval = CONCAT_(func,LDBL_SUFFIX) (args);\
	   default:\
	      hypre_printf("Unknown solver precision" );\
	      exit(0);\
        }\

/* code for pointer return type */        
#define MP_METHOD_FUNCPTR_NP(rval,func,args...)\
	switch(precision) {\
	   case HYPRE_REAL_SINGLE: \
	      rval = CONCAT_(func,FLT_SUFFIX) (args);\
	   case HYPRE_REAL_DOUBLE: \
	      rval = CONCAT_(func,DBL_SUFFIX) (args);\
	   case HYPRE_REAL_LONGDOUBLE: \
	      rval = CONCAT_(func,LDBL_SUFFIX) (args);\
	   default:\
	      hypre_printf("Unknown solver precision" );\
	      exit(0);\
        }\


#endif
