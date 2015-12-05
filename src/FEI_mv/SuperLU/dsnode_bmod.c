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
 * $Revision: 1.11 $
 ***********************************************************************EHEADER*/





/*
 * -- SuperLU routine (version 2.0) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * November 15, 1997
 *
 * Changes made to this file corresponding to calls to blas/lapack functions
 * in Nov 2003 at LLNL
 */
/*
  Copyright (c) 1994 by Xerox Corporation.  All rights reserved.
 
  THIS MATERIAL IS PROVIDED AS IS, WITH ABSOLUTELY NO WARRANTY
  EXPRESSED OR IMPLIED.  ANY USE IS AT YOUR OWN RISK.
 
  Permission is hereby granted to use or copy this program for any
  purpose, provided the above notices are retained on all copies.
  Permission to modify the code and to distribute modified code is
  granted, provided the above notices are retained, and a notice that
  the code was modified is included with the above copyright notice.
*/


#ifndef HYPRE_USING_HYPRE_BLAS
#define USE_VENDOR_BLAS
#endif

#include "dsp_defs.h"
#include "superlu_util.h"

/* 
 * Function prototypes 
 */
extern int hypre_F90_NAME_BLAS(dtrsv,DTRSV)(char *, char *, char *, int *, double *, int *, 
	                                    double *, int *);
extern int hypre_F90_NAME_BLAS(dgemv,DGEMV)(char *, int *, int *, double *, double *, int *, 
		                            double *, int *, double *, double *, int *);
void sludlsolve(int, int, double*, double*);
void sludmatvec(int, int, int, double*, double*, double*);


/*
 * Performs numeric block updates within the relaxed snode. 
 */
int
dsnode_bmod (
	    const int  jcol,	  /* in */
	    const int  jsupno,    /* in */
	    const int  fsupc,     /* in */
	    double     *dense,    /* in */
	    double     *tempv,    /* working array */
	    GlobalLU_t *Glu       /* modified */
	    )
{
#ifdef USE_VENDOR_BLAS
#ifdef _CRAY
    _fcd ftcs1 = _cptofcd("L", strlen("L")),
	 ftcs2 = _cptofcd("N", strlen("N")),
	 ftcs3 = _cptofcd("U", strlen("U"));
#endif
    int            incx = 1, incy = 1;
    double         alpha = -1.0, beta = 1.0;
#else
    int            i, iptr; 
#endif

    int            luptr, nsupc, nsupr, nrow;
    int            isub, irow; 
    register int   ufirst, nextlu;
    int            *lsub, *xlsub;
    double         *lusup;
    int            *xlusup;
    extern SuperLUStat_t SuperLUStat;
    flops_t *ops = SuperLUStat.ops;

    lsub    = Glu->lsub;
    xlsub   = Glu->xlsub;
    lusup   = Glu->lusup;
    xlusup  = Glu->xlusup;

    nextlu = xlusup[jcol];
    
    /*
     *	Process the supernodal portion of L\U[*,j]
     */
    for (isub = xlsub[fsupc]; isub < xlsub[fsupc+1]; isub++) {
  	irow = lsub[isub];
	lusup[nextlu] = dense[irow];
	dense[irow] = 0;
	++nextlu;
    }

    xlusup[jcol + 1] = nextlu;	/* Initialize xlusup for next column */
    
    if ( fsupc < jcol ) {

	luptr = xlusup[fsupc];
	nsupr = xlsub[fsupc+1] - xlsub[fsupc];
	nsupc = jcol - fsupc;	/* Excluding jcol */
	ufirst = xlusup[jcol];	/* Points to the beginning of column
				   jcol in supernode L\U(jsupno). */
	nrow = nsupr - nsupc;

	ops[TRSV] += nsupc * (nsupc - 1);
	ops[GEMV] += 2 * nrow * nsupc;

#ifdef USE_VENDOR_BLAS
#ifdef _CRAY
	STRSV( ftcs1, ftcs2, ftcs3, &nsupc, &lusup[luptr], &nsupr, 
	      &lusup[ufirst], &incx );
	SGEMV( ftcs2, &nrow, &nsupc, &alpha, &lusup[luptr+nsupc], &nsupr, 
		&lusup[ufirst], &incx, &beta, &lusup[ufirst+nsupc], &incy );
#else
	hypre_F90_NAME_BLAS(dtrsv,DTRSV)( "L", "N", "U", &nsupc, &lusup[luptr], &nsupr, 
	      &lusup[ufirst], &incx );
	hypre_F90_NAME_BLAS(dgemv,DGEMV)( "N", &nrow, &nsupc, &alpha, &lusup[luptr+nsupc], &nsupr, 
		&lusup[ufirst], &incx, &beta, &lusup[ufirst+nsupc], &incy );
#endif
#else
	sludlsolve ( nsupr, nsupc, &lusup[luptr], &lusup[ufirst] );
	sludmatvec ( nsupr, nrow, nsupc, &lusup[luptr+nsupc], 
			&lusup[ufirst], &tempv[0] );

        /* Scatter tempv[*] into lusup[*] */
	iptr = ufirst + nsupc;
	for (i = 0; i < nrow; i++) {
	    lusup[iptr++] -= tempv[i];
	    tempv[i] = 0.0;
	}
#endif

    }

    return 0;
}
