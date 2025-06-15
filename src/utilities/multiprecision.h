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
* For mixed-precision build only
*---------------------------------------------------------------------------*/

#if defined(HYPRE_MIXED_PRECISION)

/* object precision options */
typedef enum 
{
   HYPRE_REAL_SINGLE,
   HYPRE_REAL_DOUBLE,
   HYPRE_REAL_LONGDOUBLE

} HYPRE_Precision;

/* Set build options */
#if defined(MP_BUILD_SINGLE)

#define HYPRE_MULTIPRECISION_FUNC(a) CONCAT_(a, FLT_SUFFIX)
#define HYPRE_ZZZZZPRECISION_FUNC(a) CONCAT_(a, FLT_SUFFIX)  /* RDF: Temporary */
#define HYPRE_FIXEDPRECISION_FUNC(a) CONCAT_(a, FLT_SUFFIX)
#define HYPRE_OBJECT_PRECISION HYPRE_REAL_SINGLE
#define BUILD_MP_FUNC 1
#ifndef HYPRE_SINGLE
#define HYPRE_SINGLE 1
#endif
#undef  HYPRE_LONG_DOUBLE

#elif defined(MP_BUILD_LONGDOUBLE)

#define HYPRE_MULTIPRECISION_FUNC(a) CONCAT_(a, LDBL_SUFFIX)
#define HYPRE_ZZZZZPRECISION_FUNC(a) CONCAT_(a, LDBL_SUFFIX)  /* RDF: Temporary */
#define HYPRE_FIXEDPRECISION_FUNC(a) CONCAT_(a, LDBL_SUFFIX)
#define HYPRE_OBJECT_PRECISION HYPRE_REAL_LONGDOUBLE
#define BUILD_MP_FUNC 1
#undef  HYPRE_SINGLE
#ifndef HYPRE_LONG_DOUBLE
#define HYPRE_LONG_DOUBLE 1
#endif

#elif defined(MP_BUILD_DOUBLE)

#define HYPRE_MULTIPRECISION_FUNC(a) CONCAT_(a, DBL_SUFFIX)
#define HYPRE_ZZZZZPRECISION_FUNC(a) CONCAT_(a, DBL_SUFFIX)  /* RDF: Temporary */
#define HYPRE_FIXEDPRECISION_FUNC(a) CONCAT_(a, DBL_SUFFIX)
#define HYPRE_OBJECT_PRECISION HYPRE_REAL_DOUBLE
#define BUILD_MP_FUNC 1
#undef  HYPRE_SINGLE
#undef  HYPRE_LONG_DOUBLE

#elif defined(MP_BUILD_FIXED)

#define HYPRE_MULTIPRECISION_FUNC(a) a
#define HYPRE_ZZZZZPRECISION_FUNC(a) CONCAT_(a, fixed)  /* RDF: Temporary */
#define HYPRE_FIXEDPRECISION_FUNC(a) a
#define HYPRE_OBJECT_PRECISION HYPRE_REAL_DOUBLE  /* RDF: Set this to default precision */
#define BUILD_MP_FUNC 1
#define DEFINE_GLOBAL_VARIABLE 1  /* Define globals only once during fixed precision build */

#else

#define HYPRE_MULTIPRECISION_FUNC(a) a
#define HYPRE_ZZZZZPRECISION_FUNC(a) a  /* RDF: Temporary */
#define HYPRE_FIXEDPRECISION_FUNC(a) a
#define HYPRE_OBJECT_PRECISION HYPRE_REAL_DOUBLE  /* RDF: Set this to default precision */
#define DEFINE_GLOBAL_VARIABLE 1  /* RDF: Delete this later */

#endif

// /*--------------------------------------------------------------------------
// * Reset build types for Multi-precision build
// *---------------------------------------------------------------------------*/
// #if defined(MP_BUILD_SINGLE)
// #define HYPRE_OBJECT_PRECISION HYPRE_REAL_SINGLE
// #define BUILD_MP_FUNC 1
// #undef HYPRE_LONG_DOUBLE
// #ifndef HYPRE_SINGLE
// #define HYPRE_SINGLE 1
// #endif
// #elif defined(MP_BUILD_LONGDOUBLE)
// #define HYPRE_OBJECT_PRECISION HYPRE_REAL_LONGDOUBLE
// #define BUILD_MP_FUNC 1
// #undef HYPRE_SINGLE
// #ifndef HYPRE_LONG_DOUBLE
// #define HYPRE_LONG_DOUBLE 1
// #endif
// #elif defined(MP_BUILD_DOUBLE)
// #define HYPRE_OBJECT_PRECISION HYPRE_REAL_DOUBLE
// #define BUILD_MP_FUNC 1
// #undef HYPRE_SINGLE
// #undef HYPRE_LONG_DOUBLE
// #else
// /* Set a default precision */
// #define HYPRE_OBJECT_PRECISION HYPRE_REAL_DOUBLE
// /* Define globals only once during default build */
// #define DEFINE_GLOBAL_VARIABLE 1
// #ifdef BUILD_MP_FUNC
// #undef BUILD_MP_FUNC
// #endif
// #define BUILD_NON_MP_FUNC 1
// #endif
// /*--------------------------------------------------------------------------
//  * HYPRE multiprecision extensions
//  *--------------------------------------------------------------------------*/
// /* Macro to generate typed functions */
// #if defined(BUILD_MP_FUNC)
// #if defined(HYPRE_SINGLE)
// //#define FUNC_SUFFIX flt
// #define HYPRE_MULTIPRECISION_FUNC(a) CONCAT_(a, FLT_SUFFIX)
// #elif defined(HYPRE_LONG_DOUBLE)
// //#define FUNC_SUFFIX long_dbl
// #define HYPRE_MULTIPRECISION_FUNC(a) CONCAT_(a, LDBL_SUFFIX)
// #else /* HYPRE_DOUBLE */
// //#define FUNC_SUFFIX dbl
// #define HYPRE_MULTIPRECISION_FUNC(a) CONCAT_(a, DBL_SUFFIX)
// #endif
// #else
// #define HYPRE_MULTIPRECISION_FUNC(a) a
// #endif

/* Apply suffix to define typed function */
//#define HYPRE_MULTIPRECISION_FUNC(a) CONCAT_(a, FUNC_SUFFIX)

#else
/* define no-op for typed function macro */
//#define HYPRE_MULTIPRECISION_FUNC(a) a
#define BUILD_MP_FUNC 1
#define BUILD_NON_MP_FUNC 1
#define DEFINE_GLOBAL_VARIABLE 1
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

