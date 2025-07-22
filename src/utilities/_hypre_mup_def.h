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
 * NOTE: This header is needed in both user and internal header contexts, even
 * though most of the definitions are not intended for users.  Because of this,
 * a mix of capital and lower-case hypre prefixes are used.
 *****************************************************************************/
 
#ifndef MULTIPRECISION_UTILITIES_HEADER
#define MULTIPRECISION_UTILITIES_HEADER

#define hypre_CONCAT2_(a, b) a ## _ ## b
#define hypre_CONCAT_(a, b) hypre_CONCAT2_(a, b)

/* multiprecision build types - RDF: These don't buy us much since they are only
 * used below.  We should consider removing them for simplicity.  That will also
 * eliminate the need for the CONCAT functions above. */
#define hypre_FLT_SUFFIX flt
#define hypre_DBL_SUFFIX dbl
#define hypre_LDBL_SUFFIX long_dbl

/*--------------------------------------------------------------------------
 * For mixed-precision build only
 *---------------------------------------------------------------------------*/

#if defined(HYPRE_MIXED_PRECISION)

/* Set build options */
#if defined(MP_BUILD_SINGLE)

#define hypre_MP_BUILD 1
#define HYPRE_MULTIPRECISION_FUNC(a) hypre_CONCAT_(a, hypre_FLT_SUFFIX)
#define HYPRE_ZZZZZPRECISION_FUNC(a) hypre_CONCAT_(a, hypre_FLT_SUFFIX)  /* RDF: Temporary */
#define HYPRE_FIXEDPRECISION_FUNC(a) hypre_CONCAT_(a, hypre_FLT_SUFFIX)
#undef  HYPRE_LONG_DOUBLE
#ifndef HYPRE_SINGLE
#define HYPRE_SINGLE 1
#endif

#elif defined(MP_BUILD_DOUBLE)

#define hypre_MP_BUILD 1
#define HYPRE_MULTIPRECISION_FUNC(a) hypre_CONCAT_(a, hypre_DBL_SUFFIX)
#define HYPRE_ZZZZZPRECISION_FUNC(a) hypre_CONCAT_(a, hypre_DBL_SUFFIX)  /* RDF: Temporary */
#define HYPRE_FIXEDPRECISION_FUNC(a) hypre_CONCAT_(a, hypre_DBL_SUFFIX)
#undef  HYPRE_SINGLE
#undef  HYPRE_LONG_DOUBLE
#define hypre_DEFINE_GLOBAL_MP 1  /* Define globals only once during double precision build */

#elif defined(MP_BUILD_LONGDOUBLE)

#define hypre_MP_BUILD 1
#define HYPRE_MULTIPRECISION_FUNC(a) hypre_CONCAT_(a, hypre_LDBL_SUFFIX)
#define HYPRE_ZZZZZPRECISION_FUNC(a) hypre_CONCAT_(a, hypre_LDBL_SUFFIX)  /* RDF: Temporary */
#define HYPRE_FIXEDPRECISION_FUNC(a) hypre_CONCAT_(a, hypre_LDBL_SUFFIX)
#undef  HYPRE_SINGLE
#ifndef HYPRE_LONG_DOUBLE
#define HYPRE_LONG_DOUBLE 1
#endif

#elif defined(MP_BUILD_DEFAULT)

#define hypre_MP_BUILD 1
#define HYPRE_MULTIPRECISION_FUNC(a) a
#define HYPRE_ZZZZZPRECISION_FUNC(a) hypre_CONCAT_(a, def)  /* RDF: Temporary */
#define HYPRE_FIXEDPRECISION_FUNC(a) a

#else

#define HYPRE_MULTIPRECISION_FUNC(a) a
#define HYPRE_ZZZZZPRECISION_FUNC(a) hypre_CONCAT_(a, def)  /* RDF: Temporary */
#define HYPRE_FIXEDPRECISION_FUNC(a) a

#endif

#endif

#if 0
/* Helper macros to generate multiprecision function declarations */
#define DECLARE_MP_FUNC(rtype,func,fargs...)\
	rtype CONCAT_(func,FLT_SUFFIX) (fargs);\
	rtype CONCAT_(func,DBL_SUFFIX) (fargs);\
	rtype CONCAT_(func,LDBL_SUFFIX) (fargs);

#define DECLARE_DP_FUNC(rtype,func,fargs...)\
	rtype CONCAT_(func,DBL_SUFFIX) (fargs);

#define HYPRE_DP_FUNC(a) CONCAT_(a, DBL_SUFFIX)

#define DECLARE_SP_FUNC(rtype,func,fargs...)\
	rtype CONCAT_(func,FLT_SUFFIX) (fargs);

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
        }

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
        }

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
        }
#endif

#endif

