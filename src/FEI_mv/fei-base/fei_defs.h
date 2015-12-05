
#ifndef _fei_defs_h_
#define _fei_defs_h_

/*--------------------------------------------------------------------*/
/*    Copyright 2005 Sandia Corporation.                              */
/*    Under the terms of Contract DE-AC04-94AL85000, there is a       */
/*    non-exclusive license for use of this work by or on behalf      */
/*    of the U.S. Government.  Export of this program may require     */
/*    a license from the United States Government.                    */
/*--------------------------------------------------------------------*/

/*
   In this file we set some #defines to use as parameters to
   some fei functions, and also some error-code returns.
   We also provide the typedef for 'GlobalID' which appears in
   many FEI function prototypes. Note that the default case is
   for GlobalID to simply be an int.
   This file is included by both C and C++ versions of the fei.
*/

#ifdef EIGHT_BYTE_GLOBAL_ID
typedef long long   GlobalID;
#define GlobalID_MAX LLONG_MAX
#define GlobalID_MIN LLONG_MIN
#else
    typedef int GlobalID;
#endif


/* solveType (used in 'setSolveType'): */
#define FEI_SINGLE_SYSTEM     0
#define FEI_EIGEN_SOLVE       1
#define FEI_AGGREGATE_SUM     2
#define FEI_AGGREGATE_PRODUCT 3

/* IDType (used in coefficient-access functions) */
#define FEI_NODE 0
#define FEI_ELEMENT 1
#define FEI_ONLY_NODES 2
#define FEI_ONLY_ELEMENTS 3

/* elemFormat (used in 'sumInElem' and 'sumInElemMatrix'): */
#define FEI_DENSE_ROW      0
#define FEI_UPPER_SYMM_ROW 1
#define FEI_LOWER_SYMM_ROW 2
#define FEI_DENSE_COL      3
#define FEI_UPPER_SYMM_COL 4
#define FEI_LOWER_SYMM_COL 5
#define FEI_DIAGONAL       6


/* interleaveStrategy (used in initElemBlock): */
#define FEI_NODE_MAJOR  0
#define FEI_FIELD_MAJOR 1

/* FEI function return values */
#define FEI_SUCCESS         0
#define FEI_FATAL_ERROR    -1
#define FEI_ID_NOT_FOUND   -2

#endif

