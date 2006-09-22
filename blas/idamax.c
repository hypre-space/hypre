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




/*  -- translated by f2c (version 19940927).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

#include "f2c.h"
#include "hypre_blas.h"

integer idamax_(integer *n, doublereal *dx, integer *incx)
{


    /* System generated locals */
    integer ret_val;
    doublereal d__1;

    /* Local variables */
    static doublereal dmax__;
    static integer i, ix;


/*     finds the index of element having max. absolute value.   
       jack dongarra, linpack, 3/11/78.   
       modified 3/93 to return if incx .le. 0.   
       modified 12/3/93, array(1) declarations changed to array(*)   


    
   Parameter adjustments   
       Function Body */
#define DX(I) dx[(I)-1]


    ret_val = 0;
    if (*n < 1 || *incx <= 0) {
	return ret_val;
    }
    ret_val = 1;
    if (*n == 1) {
	return ret_val;
    }
    if (*incx == 1) {
	goto L20;
    }

/*        code for increment not equal to 1 */

    ix = 1;
    dmax__ = abs(DX(1));
    ix += *incx;
    for (i = 2; i <= *n; ++i) {
	if ((d__1 = DX(ix), abs(d__1)) <= dmax__) {
	    goto L5;
	}
	ret_val = i;
	dmax__ = (d__1 = DX(ix), abs(d__1));
L5:
	ix += *incx;
/* L10: */
    }
    return ret_val;

/*        code for increment equal to 1 */

L20:
    dmax__ = abs(DX(1));
    for (i = 2; i <= *n; ++i) {
	if ((d__1 = DX(i), abs(d__1)) <= dmax__) {
	    goto L30;
	}
	ret_val = i;
	dmax__ = (d__1 = DX(i), abs(d__1));
L30:
	;
    }
    return ret_val;
} /* idamax_ */

