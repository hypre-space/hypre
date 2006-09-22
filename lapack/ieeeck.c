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

integer ieeeck_(integer *ispec, real *zero, real *one)
{
/*  -- LAPACK auxiliary routine (version 3.0) --   
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,   
       Courant Institute, Argonne National Lab, and Rice University   
       June 30, 1998   


    Purpose   
    =======   

    IEEECK is called from the ILAENV to verify that Infinity and   
    possibly NaN arithmetic is safe (i.e. will not trap).   

    Arguments   
    =========   

    ISPEC   (input) INTEGER   
            Specifies whether to test just for inifinity arithmetic   
            or whether to test for infinity and NaN arithmetic.   
            = 0: Verify infinity arithmetic only.   
            = 1: Verify infinity and NaN arithmetic.   

    ZERO    (input) REAL   
            Must contain the value 0.0   
            This is passed to prevent the compiler from optimizing   
            away this code.   

    ONE     (input) REAL   
            Must contain the value 1.0   
            This is passed to prevent the compiler from optimizing   
            away this code.   

    RETURN VALUE:  INTEGER   
            = 0:  Arithmetic failed to produce the correct answers   
            = 1:  Arithmetic produced the correct answers */
    /* System generated locals */
    integer ret_val;
    /* Local variables */
    static real neginf, posinf, negzro, newzro, nan1, nan2, nan3, nan4, nan5, 
	    nan6;


    ret_val = 1;

    posinf = *one / *zero;
    if (posinf <= *one) {
	ret_val = 0;
	return ret_val;
    }

    neginf = -(*one) / *zero;
    if (neginf >= *zero) {
	ret_val = 0;
	return ret_val;
    }

    negzro = *one / (neginf + *one);
    if (negzro != *zero) {
	ret_val = 0;
	return ret_val;
    }

    neginf = *one / negzro;
    if (neginf >= *zero) {
	ret_val = 0;
	return ret_val;
    }

    newzro = negzro + *zero;
    if (newzro != *zero) {
	ret_val = 0;
	return ret_val;
    }

    posinf = *one / newzro;
    if (posinf <= *one) {
	ret_val = 0;
	return ret_val;
    }

    neginf *= posinf;
    if (neginf >= *zero) {
	ret_val = 0;
	return ret_val;
    }

    posinf *= posinf;
    if (posinf <= *one) {
	ret_val = 0;
	return ret_val;
    }




/*     Return if we were only asked to check infinity arithmetic */

    if (*ispec == 0) {
	return ret_val;
    }

    nan1 = posinf + neginf;

    nan2 = posinf / neginf;

    nan3 = posinf / posinf;

    nan4 = posinf * *zero;

    nan5 = neginf * negzro;

    nan6 = nan5 * 0.f;

    if (nan1 == nan1) {
	ret_val = 0;
	return ret_val;
    }

    if (nan2 == nan2) {
	ret_val = 0;
	return ret_val;
    }

    if (nan3 == nan3) {
	ret_val = 0;
	return ret_val;
    }

    if (nan4 == nan4) {
	ret_val = 0;
	return ret_val;
    }

    if (nan5 == nan5) {
	ret_val = 0;
	return ret_val;
    }

    if (nan6 == nan6) {
	ret_val = 0;
	return ret_val;
    }

    return ret_val;
} /* ieeeck_ */

