/*
 * File:        sidlAsserts.h
 * Revision:    @(#) $Revision$
 * Date:        $Date$
 * Description: convenience C macros for managing SIDL Assertions
 *
 * Copyright (c) 2004, The Regents of the University of Calfornia.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the Components Team <components@llnl.gov>
 * UCRL-CODE-2002-054
 * All rights reserved.
 * 
 * This file is part of Babel. For more information, see
 * http://www.llnl.gov/CASC/components/. Please read the COPYRIGHT file
 * for Our Notice and the LICENSE file for the GNU Lesser General Public
 * License.
 * 
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License (as published by
 * the Free Software Foundation) version 2.1 dated February 1999.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
 * conditions of the GNU Lesser General Public License for more details.
 * 
 * You should have recieved a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 ****************************************************************************
 * WARNINGS:
 * 1) SIDL_FULL_ASTATS
 *    This macro is used here to determine whether a full set of statistics
 *    are going to be employed or only the bare minimum.  It is assumed that
 *    this macro is kept in sync with the contents of S_FULL_STATS_MACRO in 
 *    IOR.java.
 ****************************************************************************
 *
 * The following include files are needed:
 *    math.h                   For the ceiling function used by the 
 *                               random and timing-based policies.
 *    stdlib.h                 For random number generation (including 
 *                               RAND_MAX).
 *    time.h                   For processing associated with the 
 *                               timing-based policy.
 *    sidl_PreViolation.h   
 *    sidl_PostViolation.h 
 *    sidl_InvViolation.h      For ease-of-use since this (single) header
 *                               is currently providing both the IOR
 *                               and (C) applications with enforcement
 *                               options.
 */

#ifndef included_sidlAsserts_h
#define included_sidlAsserts_h
#include <math.h>
#include <sys/time.h>
#include <stdlib.h>
#ifndef included_sidl_PreViolation_h
#include "sidl_PreViolation.h"
#endif
#ifndef included_sidl_PostViolation_h
#include "sidl_PostViolation.h"
#endif
#ifndef included_sidl_InvViolation_h
#include "sidl_InvViolation.h"
#endif

/****************************************************************************
 * SIDL Assertion Checking option, or level, masks
 *
 * There are three types of assertions supported for interfaces and five 
 * frequencies that can be combined with an optional adaptation.
 *
 * The assertion types are:
 *      preconditions,
 *      postconditions, and
 *      invariants.
 * The first set of masks are used to specify combinations of these checks.
 * You can "or" together the desired combination of these types OR simply
 * use one of the combinations given for your convenience.
 *
 * The five basic frequency, or sampling, options are:
 *      always       => always check the specified type(s) of assertions
 *      periodically => only check the assertions on a periodic basis
 *      proportional => check assertions as often as possibly such that
 *                      the amount of time spent checking adds no more than
 *                      a user-defined fraction of the time spent executing
 *                      the method dispatch.
 *                      (WARNING: This needs to be adjusted to factor in the
 *                      overhead of up to three gettimeofday() calls!)
 *      randomly     => check assertions on a random basis
 *
 * In all cases, the assertions are checked the first time the method
 * is invoked.  As long as assertions fail for a method, regardless of
 * frequency option, it's assertions will be checked upon invocation
 * until such time as the assertions pass.  
 *
 * NOTE that an alternative checking frequency is NEVER, which means that
 * assertions are never checked.  In fact, the annotated functions are 
 * completely by-passed.
 ****************************************************************************/

/*
 * ... Basic assertion types to be checked
 */
#ifndef CHECK_NO_TYPES
#define CHECK_NO_TYPES       0
#endif
#ifndef CHECK_PRECONDITIONS
#define CHECK_PRECONDITIONS  1
#endif
#ifndef CHECK_POSTCONDITIONS
#define CHECK_POSTCONDITIONS 2
#endif
#ifndef CHECK_INVARIANTS
#define CHECK_INVARIANTS     4
#endif

/*
 * ...... PLUS combinations of basic assertion types to be checked
 */
#ifndef CHECK_PRE_POST_ONLY
#define CHECK_PRE_POST_ONLY  3
#endif
#ifndef CHECK_PRE_INV_ONLY
#define CHECK_PRE_INV_ONLY   5
#endif
#ifndef CHECK_POST_INV_ONLY
#define CHECK_POST_INV_ONLY  6
#endif
#ifndef CHECK_ALL_TYPES
#define CHECK_ALL_TYPES      7
#endif

/*
 * ... Adaptive checking enforcement
 */
#ifndef CHECK_ADAPTIVELY
#define CHECK_ADAPTIVELY     8
#endif
 
/*
 * ... Basic checking frequency levels
 */

#ifndef CHECK_ALWAYS
#define CHECK_ALWAYS         16
#endif
#ifndef CHECK_PERIODICALLY
#define CHECK_PERIODICALLY   32
#endif
#ifndef CHECK_TIMING
#define CHECK_TIMING         64
#endif
#ifndef CHECK_RANDOMLY
#define CHECK_RANDOMLY       128
#endif

/*
 * ...... PLUS a mask for determining if any FREQUENCY is specified (e.g.,
 *        (MY_CHECKING & CHECK_ASSERTIONS) will be TRUE if (at least) one of 
 *        the options is set in MY_CHECKING) -- though ONLY one should be!
 */
#ifndef CHECK_ASSERTIONS
#define CHECK_ASSERTIONS     240
#endif


/****************************************************************************
 * SIDL Assertion macros
 ****************************************************************************/

/*
 * SIDL_ARRAY_ALL_BOTH	all(a1 r a2), where a1 and a2 are arrays, r is the
 *                      relation
 */
#define SIDL_ARRAY_ALL_BOTH(AC1, AV1, AC2, AV2, REL, I, NUM, C, BRES) \
   SIDL_ARRAY_COUNT_BOTH(AC1, AV1, AC2, AV2, REL, I, NUM, C);\
   BRES = (C == NUM);\
}

/*
 * SIDL_ARRAY_ALL_LR	all(vr a), where a is array, vr is value + relation
 */
#define SIDL_ARRAY_ALL_LR(AC, AV, REL, I, NUM, C, BRES) {\
   SIDL_ARRAY_COUNT_LR(AC, AV, REL, I, NUM, C);\
   BRES = (C == NUM);\
}

/*
 * SIDL_ARRAY_ALL_RR	all(a rv), where a is array, rv is relation + value
 */
#define SIDL_ARRAY_ALL_RR(AC, AV, REL, I, NUM, C, BRES) {\
   SIDL_ARRAY_COUNT_RR(AC, AV, REL, I, NUM, C);\
   BRES = (C == NUM);\
}

/*
 * SIDL_ARRAY_ANY_BOTH	any(a1 r a2), where a1 and a2 are arrays, r is the
 *                      relation
 *
 *   NOTE: Will return FALSE if the arrays are not the same size.
 */
#define SIDL_ARRAY_ANY_BOTH(AC1, AV1, AC2, AV2, REL, I, NUM, BRES) {\
   BRES = FALSE;\
   NUM  = SIDL_ARRAY_SIZE(AC1, (AV1));\
   if (SIDL_ARRAY_SIZE(AC2, (AV2)) == NUM) {\
     for (I=0; (I<NUM) && (!BRES); I++) {\
       SIDL_INCR_IF_TRUE((AC1##_get1((AV1),I) REL AC2##_get1((AV2),I)), BRES)\
     }\
   }\
}

/*
 * SIDL_ARRAY_ANY_LR	any(vr a), where a is array, vr is value + relation
 */
#define SIDL_ARRAY_ANY_LR(AC, AV, REL, I, NUM, BRES) {\
   BRES = FALSE;\
   NUM  = SIDL_ARRAY_SIZE(AC, (AV));\
   for (I=0; (I<NUM) && (!BRES); I++) {\
     SIDL_INCR_IF_TRUE((REL AC##_get1((AV),I)), BRES)\
   }\
}

/*
 * SIDL_ARRAY_ANY_RR	any(a rv), where a is array, rv is relation + value
 */
#define SIDL_ARRAY_ANY_RR(AC, AV, REL, I, NUM, BRES) {\
   BRES = FALSE;\
   NUM  = SIDL_ARRAY_SIZE(AC, (AV));\
   for (I=0; (I<NUM) && (!BRES); I++) {\
     SIDL_INCR_IF_TRUE((AC##_get1((AV),I) REL), BRES)\
   }\
}

/*
 * SIDL_ARRAY_COUNT_BOTH  count(a1 r a2), where a1 and a2 are arrays, r is 
 *                        the relation.
 *
 *   NOTE: Will return FALSE if the arrays are not the same size.
 */
#define SIDL_ARRAY_COUNT_BOTH(AC1, AV1, AC2, AV2, REL, I, NUM, IRES) {\
   IRES = 0;\
   NUM = SIDL_ARRAY_SIZE(AC1, (AV1));\
   if (SIDL_ARRAY_SIZE(AC2, (AV2)) == NUM) {\
     for (I=0; I<NUM; I++) {\
       SIDL_INCR_IF_TRUE((AC1##_get1((AV1),I) REL AC2##_get1((AV2),I)), IRES)\
     }\
   }\
}

/*
 * SIDL_ARRAY_COUNT_LR	count(vr a), where a is array, vr is value + relation
 */
#define SIDL_ARRAY_COUNT_LR(AC, AV, REL, I, NUM, IRES) {\
   IRES = 0;\
   NUM = SIDL_ARRAY_SIZE(AC, (AV));\
   for (I=0; I<NUM; I++) {\
     SIDL_INCR_IF_TRUE((REL AC##_get1((AV),I)), IRES)\
   }\
}

/*
 * SIDL_ARRAY_COUNT_RR	count(a rv), where a is array, rv is relation + value
 */
#define SIDL_ARRAY_COUNT_RR(AC, AV, REL, I, NUM, IRES) {\
   IRES = 0;\
   NUM = SIDL_ARRAY_SIZE(AC, (AV));\
   for (I=0; I<NUM; I++) {\
     SIDL_INCR_IF_TRUE((AC##_get1((AV),I) REL), IRES)\
   }\
}

/*
 * SIDL_ARRAY_DIMEN	dimen(a), where a is the array
 */
#define SIDL_ARRAY_DIMEN(AC, AV) AC##_dimen(AV)

/*
 * SIDL_ARRAY_IRANGE	irange(a, v1, v2), where a is array whose integer
 *                      values are to be in v1..v2.
 */
#define SIDL_ARRAY_IRANGE(AC, AV, V1, V2, I, NUM, C, BRES) {\
   C   = 0;\
   NUM = SIDL_ARRAY_SIZE(AC, (AV));\
   for (I=0; I<NUM; I++) {\
     SIDL_INCR_IF_TRUE(\
       SIDL_IRANGE((double)AC##_get1((AV),I), (double)V1, (double)V2), C)\
   }\
   BRES = (C == NUM);\
}

/*
 * SIDL_ARRAY_LOWER	lower(a, d), where a is the array and d is the dimension
 */
#define SIDL_ARRAY_LOWER(AC, AV, D) AC##_lower((AV), D)

/*
 * SIDL_ARRAY_MAX	max(a), where a is the array of scalar
 */
#define SIDL_ARRAY_MAX(AC, AV, I, NUM, T, RES) {\
   RES  = AC##_get1((AV),0);\
   NUM = SIDL_ARRAY_SIZE(AC, (AV));\
   for (I=1; I<NUM; I++) {\
     T _SAMAXV = AC##_get1((AV),I);\
     if (_SAMAXV > RES) { RES = _SAMAXV; } \
   }\
}

/*
 * SIDL_ARRAY_MIN	min(a), where a is the array of scalar
 */
#define SIDL_ARRAY_MIN(AC, AV, I, NUM, T, RES) {\
   RES  = AC##_get1((AV),0);\
   NUM = SIDL_ARRAY_SIZE(AC, (AV));\
   for (I=1; I<NUM; I++) {\
     T _SAMINV = AC##_get1((AV),I);\
     if (_SAMINV < RES) { RES = _SAMINV; } \
   }\
}

/*
 * SIDL_ARRAY_NEAR_EQUAL	nearEqual(a, b, tol), where a and b are arrays
 *                              whose scalar values are to be compared.
 */
#define SIDL_ARRAY_NEAR_EQUAL(AC1, AV1, AC2, AV2, TOL, I, NUM, C, BRES) {\
   C = 0;\
   NUM = SIDL_ARRAY_SIZE(AC1, (AV1));\
   for (I=0; I<NUM; I++) {\
     SIDL_INCR_IF_TRUE(\
       SIDL_NEAR_EQUAL(AC1##_get1((AV1),I), AC2##_get1((AV2),I), TOL), C)\
   }\
   BRES = (C == NUM);\
}

/*
 * SIDL_ARRAY_NON_INCR 	nonIncr(a), where a is array of numeric values
 *                      to be checked for being in decreasing order.
 */
#define SIDL_ARRAY_NON_INCR(AC, AV, I, NUM, V, BRES) {\
   BRES = TRUE;\
   V    = ((AV) != NULL) ? (double) AC##_get1((AV),0) : 0.0;\
   NUM  = SIDL_ARRAY_SIZE(AC, (AV));\
   for (I=0; (I<NUM) && (BRES); I++) {\
     if ((double)AC##_get1((AV),I) > V) {\
       BRES = FALSE; \
     } else {\
       V = (double) AC##_get1((AV),0);\
     }\
   }\
}

/*
 * SIDL_ARRAY_NONE_BOTH		none(a1 r a2), where a1 and a2 are arrays, r is
 *                       	the relation.
 */
#define SIDL_ARRAY_NONE_BOTH(AC1, AV1, AC2, AV2, REL, I, NUM, C, BRES) {\
   SIDL_ARRAY_COUNT_BOTH(AC1, AV1, AC2, AV2, REL, I, NUM, C);\
   BRES = (C == 0);\
}

/*
 * SIDL_ARRAY_NONE_LR	none(vr a), where a is array, vr is value + relation
 */
#define SIDL_ARRAY_NONE_LR(AC, AV, REL, I, NUM, C, BRES) {\
   SIDL_ARRAY_COUNT_LR(AC, AV, REL, I, NUM, C);\
   BRES = (C == 0);\
}

/*
 * SIDL_ARRAY_NONE_RR	none(a rv), where a is array, rv is relation + value
 */
#define SIDL_ARRAY_NONE_RR(AC, AV, REL, I, NUM, C, BRES) {\
   SIDL_ARRAY_COUNT_RR(AC, AV, REL, I, NUM, C);\
   BRES = (C == 0);\
}

/*
 * SIDL_ARRAY_RANGE	range(a, v1, v2, tol), where a is array whose scalar
 *                      values are to be in v1..v2 within tolerance tol.
 */
#define SIDL_ARRAY_RANGE(AC, AV, V1, V2, TOL, I, NUM, C, BRES) {\
   C   = 0;\
   NUM = SIDL_ARRAY_SIZE(AC, (AV));\
   for (I=0; I<NUM; I++) {\
     SIDL_INCR_IF_TRUE(\
       SIDL_RANGE((double)AC##_get1((AV),I), (double)V1, (double)V2, TOL), C)\
   }\
   BRES = (C == NUM);\
}

/*
 * SIDL_ARRAY_SIZE	size(a), where a is the array 
 */
#define SIDL_ARRAY_SIZE(AC, AV) AC##_length(AV, 0)

/*
 * SIDL_ARRAY_STRIDE	stride(a, d), where a is the array and d is the 
 *                      dimension
 */
#define SIDL_ARRAY_STRIDE(AC, AV, D) AC##_stride(AV, D)

/*
 * SIDL_ARRAY_SUM	sum(a), where a is the array of scalar
 */
#define SIDL_ARRAY_SUM(AC, AV, I, NUM, RES) {\
   RES = AC##_get1((AV),0);\
   NUM = SIDL_ARRAY_SIZE(AC, (AV));\
   for (I=1; I<NUM; I++) { RES += AC##_get1((AV),I); }\
}

/*
 * SIDL_ARRAY_NON_DECR	nonDecr(a), where a is array of numeric values
 *                      to be checked for being in increasing order.
 */
#define SIDL_ARRAY_NON_DECR(AC, AV, I, NUM, V, BRES) {\
   BRES = TRUE;\
   V    = ((AV) != NULL) ? (double) AC##_get1((AV),0) : 0.0;\
   NUM  = SIDL_ARRAY_SIZE(AC, (AV));\
   for (I=0; (I<NUM) && (BRES); I++) {\
     if ((double)AC##_get1((AV),I) < V) {\
       BRES = FALSE; \
     } else {\
       V = (double) AC##_get1((AV),0);\
     }\
   }\
}

/*
 * SIDL_ARRAY_UPPER	upper(a, d), where a is the array and d is the dimension
 */
#define SIDL_ARRAY_UPPER(AC, AV, D) AC##_upper(AV, D)

/*
 * SIDL_IRANGE		irange(v, v1, v2), where determine if v in the 
 *                      range v1..v2.
 */
#define SIDL_IRANGE(V, V1, V2) \
   (  ((double)V1 <= (double)V) && ((double)V  <= (double)V2) ) 

/*
 * SIDL_NEAR_EQUAL	nearEqual(v1, v2, tol), where v1 and v2 are scalars 
 *                      being checked for being equal within the specified 
 *                      tolerance, tol.
 */
#define SIDL_NEAR_EQUAL(V1, V2, TOL)  \
   (fabs((double)V1 - (double)V2) <= (double)TOL)

/*
 * SIDL_RANGE		range(v, v1, v2, tol), where determine if v in
 *                      the range v1..v2, within the specified tolerance, tol.
 */
#define SIDL_RANGE(V, V1, V2, TOL) {\
   (  (((double)V1 - (double)TOL) <= (double)V) \
   && ((double)V                  <= ((double)V2 + (double)TOL)) ) \
}


/****************************************************************************
 * SIDL Assertion Enforcement Macros
 ****************************************************************************/

/*
 *  SIDL_DIFF_MICROSECONDS	"Standard" time difference
 */
#define SIDL_DIFF_MICROSECONDS(T2, T1) \
  (1.0e6*(double)(T2.tv_sec-T1.tv_sec)) + (T2.tv_usec-T1.tv_usec)

/*
 *  SIDL_INCR_IF_THEN		Increment V1 if EXPR is TRUE; otherwise,
 *                              increment V2.
 */
#define SIDL_INCR_IF_THEN(EXPR, V1, V2) \
  if (EXPR) { (V1) += 1; } else { (V2) += 1; }

/*
 *  SIDL_INCR_IF_TRUE		Increment V if EXPR is TRUE.
 */
#define SIDL_INCR_IF_TRUE(EXPR, V)  if (EXPR) { (V) += 1; } 

/*
 *  SIDL_SET_COUNTDOWN		Assertion Checking Policy Enforcement
 *
 *  Assumption(s):
 *  1) Exceptions encountered during adaptive checking imply that assertions 
 *     should ALWAYS be checked on next invocation of the method regardless of 
 *     policy.
 */
#define SIDL_SET_COUNTDOWN(LVL, R, CD, AOK, MOK, MT, TT) {\
  if ((LVL) & CHECK_ADAPTIVELY) {\
    if ((AOK) && (MOK)) {\
      SIDL_SET_POLICY_CD(LVL, R, CD, MT, TT)\
    } else {\
      (CD) = 0;\
    }\
  } else {\
    SIDL_SET_POLICY_CD(LVL, R, CD, MT, TT)\
  }\
}

/*
 *  SIDL_SET_POLICY_CD  Set the countdown based only on the policy
 *
 *  Assumption(s):  N/A
 */
#define SIDL_SET_POLICY_CD(LVL, R, CD, MT, TT) {\
  if ((LVL) & CHECK_TIMING) {\
    SIDL_SET_TIMING_CD(R, MT, TT, CD)\
  } else if ((LVL) & CHECK_PERIODICALLY) {\
    (CD) = (R);\
  } else if ((LVL) & CHECK_RANDOMLY) {\
    (CD) = ceil(((double)rand()/(RAND_MAX))*(R));\
  } else { /* CHECK_ALWAYS */ \
    (CD) = 0;\
  }\
}

/*
 *  SIDL_SET_TIMING_CD	Set the proportional checking time countdown.
 *
 *  Assumption(s):
 *  1) In the highly unlikely event that the amount of time it takes to
 *     execute a method is under 1 microsecond, the TIMING checking policy
 *     assumes the method takes 1 microsecond.  (Note that this avoids a 
 *     divide-by-zero issue in the calculation of the countdown.)
 */
#define SIDL_SET_TIMING_CD(R, MT, TT, CD) {\
  if ( ((TT-MT) == 0) || ((CD) < 0) ) {\
    (CD) = -1;\
  } else {\
    if ((MT) > 0) {\
      (CD) = ceil((((double)((TT)-(MT)))/(double)((MT)?(MT):1))*(1.0/((R)?(R):1.0))) - 1;\
    } else {\
      (CD) = ceil(((double)(TT)-1.0)*(1.0/((R)?(R):1.0))) - 1;\
    }\
  }\
}

#endif /* included_sidlAsserts_h */
