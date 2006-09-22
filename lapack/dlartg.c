/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/



#include "hypre_lapack.h"
#include "f2c.h"

/* Subroutine */ int dlartg_(doublereal *f, doublereal *g, doublereal *cs, 
	doublereal *sn, doublereal *r__)
{
/*  -- LAPACK auxiliary routine (version 3.0) --   
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,   
       Courant Institute, Argonne National Lab, and Rice University   
       September 30, 1994   


    Purpose   
    =======   

    DLARTG generate a plane rotation so that   

       [  CS  SN  ]  .  [ F ]  =  [ R ]   where CS**2 + SN**2 = 1.   
       [ -SN  CS  ]     [ G ]     [ 0 ]   

    This is a slower, more accurate version of the BLAS1 routine DROTG,   
    with the following other differences:   
       F and G are unchanged on return.   
       If G=0, then CS=1 and SN=0.   
       If F=0 and (G .ne. 0), then CS=0 and SN=1 without doing any   
          floating point operations (saves work in DBDSQR when   
          there are zeros on the diagonal).   

    If F exceeds G in magnitude, CS will be positive.   

    Arguments   
    =========   

    F       (input) DOUBLE PRECISION   
            The first component of vector to be rotated.   

    G       (input) DOUBLE PRECISION   
            The second component of vector to be rotated.   

    CS      (output) DOUBLE PRECISION   
            The cosine of the rotation.   

    SN      (output) DOUBLE PRECISION   
            The sine of the rotation.   

    R       (output) DOUBLE PRECISION   
            The nonzero component of the rotated vector.   

    ===================================================================== */
    /* Initialized data */
    static logical first = TRUE_;
    /* System generated locals */
    integer i__1;
    doublereal d__1, d__2;
    /* Builtin functions */
    double log(doublereal), pow_di(doublereal *, integer *), sqrt(doublereal);
    /* Local variables */
    static integer i__;
    static doublereal scale;
    static integer count;
    static doublereal f1, g1, safmn2, safmx2;
    extern doublereal dlamch_(char *);
    static doublereal safmin, eps;



    if (first) {
	first = FALSE_;
	safmin = dlamch_("S");
	eps = dlamch_("E");
	d__1 = dlamch_("B");
	i__1 = (integer) (log(safmin / eps) / log(dlamch_("B")) / 
		2.);
	safmn2 = pow_di(&d__1, &i__1);
	safmx2 = 1. / safmn2;
    }
    if (*g == 0.) {
	*cs = 1.;
	*sn = 0.;
	*r__ = *f;
    } else if (*f == 0.) {
	*cs = 0.;
	*sn = 1.;
	*r__ = *g;
    } else {
	f1 = *f;
	g1 = *g;
/* Computing MAX */
	d__1 = abs(f1), d__2 = abs(g1);
	scale = max(d__1,d__2);
	if (scale >= safmx2) {
	    count = 0;
L10:
	    ++count;
	    f1 *= safmn2;
	    g1 *= safmn2;
/* Computing MAX */
	    d__1 = abs(f1), d__2 = abs(g1);
	    scale = max(d__1,d__2);
	    if (scale >= safmx2) {
		goto L10;
	    }
/* Computing 2nd power */
	    d__1 = f1;
/* Computing 2nd power */
	    d__2 = g1;
	    *r__ = sqrt(d__1 * d__1 + d__2 * d__2);
	    *cs = f1 / *r__;
	    *sn = g1 / *r__;
	    i__1 = count;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		*r__ *= safmx2;
/* L20: */
	    }
	} else if (scale <= safmn2) {
	    count = 0;
L30:
	    ++count;
	    f1 *= safmx2;
	    g1 *= safmx2;
/* Computing MAX */
	    d__1 = abs(f1), d__2 = abs(g1);
	    scale = max(d__1,d__2);
	    if (scale <= safmn2) {
		goto L30;
	    }
/* Computing 2nd power */
	    d__1 = f1;
/* Computing 2nd power */
	    d__2 = g1;
	    *r__ = sqrt(d__1 * d__1 + d__2 * d__2);
	    *cs = f1 / *r__;
	    *sn = g1 / *r__;
	    i__1 = count;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		*r__ *= safmn2;
/* L40: */
	    }
	} else {
/* Computing 2nd power */
	    d__1 = f1;
/* Computing 2nd power */
	    d__2 = g1;
	    *r__ = sqrt(d__1 * d__1 + d__2 * d__2);
	    *cs = f1 / *r__;
	    *sn = g1 / *r__;
	}
	if (abs(*f) > abs(*g) && *cs < 0.) {
	    *cs = -(*cs);
	    *sn = -(*sn);
	    *r__ = -(*r__);
	}
    }
    return 0;

/*     End of DLARTG */

} /* dlartg_ */

