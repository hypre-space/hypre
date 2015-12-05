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
 * $Revision: 1.7 $
 ***********************************************************************EHEADER*/


#include "../blas/hypre_blas.h"
#include "hypre_lapack.h"
#include "f2c.h"

/* Subroutine */ int dlarf_(char *side, integer *m, integer *n, doublereal *v,
	 integer *incv, doublereal *tau, doublereal *c__, integer *ldc, 
	doublereal *work)
{
/*  -- LAPACK auxiliary routine (version 3.0) --   
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,   
       Courant Institute, Argonne National Lab, and Rice University   
       February 29, 1992   


    Purpose   
    =======   

    DLARF applies a real elementary reflector H to a real m by n matrix   
    C, from either the left or the right. H is represented in the form   

          H = I - tau * v * v'   

    where tau is a real scalar and v is a real vector.   

    If tau = 0, then H is taken to be the unit matrix.   

    Arguments   
    =========   

    SIDE    (input) CHARACTER*1   
            = 'L': form  H * C   
            = 'R': form  C * H   

    M       (input) INTEGER   
            The number of rows of the matrix C.   

    N       (input) INTEGER   
            The number of columns of the matrix C.   

    V       (input) DOUBLE PRECISION array, dimension   
                       (1 + (M-1)*abs(INCV)) if SIDE = 'L'   
                    or (1 + (N-1)*abs(INCV)) if SIDE = 'R'   
            The vector v in the representation of H. V is not used if   
            TAU = 0.   

    INCV    (input) INTEGER   
            The increment between elements of v. INCV <> 0.   

    TAU     (input) DOUBLE PRECISION   
            The value tau in the representation of H.   

    C       (input/output) DOUBLE PRECISION array, dimension (LDC,N)   
            On entry, the m by n matrix C.   
            On exit, C is overwritten by the matrix H * C if SIDE = 'L',   
            or C * H if SIDE = 'R'.   

    LDC     (input) INTEGER   
            The leading dimension of the array C. LDC >= max(1,M).   

    WORK    (workspace) DOUBLE PRECISION array, dimension   
                           (N) if SIDE = 'L'   
                        or (M) if SIDE = 'R'   

    =====================================================================   


       Parameter adjustments */
    /* Table of constant values */
    static doublereal c_b4 = 1.;
    static doublereal c_b5 = 0.;
    static integer c__1 = 1;
    
    /* System generated locals */
    integer c_dim1, c_offset;
    doublereal d__1;
    /* Local variables */
    extern /* Subroutine */ int dger_(integer *, integer *, doublereal *, 
	    doublereal *, integer *, doublereal *, integer *, doublereal *, 
	    integer *);
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int dgemv_(char *, integer *, integer *, 
	    doublereal *, doublereal *, integer *, doublereal *, integer *, 
	    doublereal *, doublereal *, integer *);


    --v;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1 * 1;
    c__ -= c_offset;
    --work;

    /* Function Body */
    if (lsame_(side, "L")) {

/*        Form  H * C */

	if (*tau != 0.) {

/*           w := C' * v */

	    dgemv_("Transpose", m, n, &c_b4, &c__[c_offset], ldc, &v[1], incv,
		     &c_b5, &work[1], &c__1);

/*           C := C - v * w' */

	    d__1 = -(*tau);
	    dger_(m, n, &d__1, &v[1], incv, &work[1], &c__1, &c__[c_offset], 
		    ldc);
	}
    } else {

/*        Form  C * H */

	if (*tau != 0.) {

/*           w := C * v */

	    dgemv_("No transpose", m, n, &c_b4, &c__[c_offset], ldc, &v[1], 
		    incv, &c_b5, &work[1], &c__1);

/*           C := C - w * v' */

	    d__1 = -(*tau);
	    dger_(m, n, &d__1, &work[1], &c__1, &v[1], incv, &c__[c_offset], 
		    ldc);
	}
    }
    return 0;

/*     End of DLARF */

} /* dlarf_ */

