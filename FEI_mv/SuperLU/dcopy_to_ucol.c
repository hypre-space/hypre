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





/*
 * -- SuperLU routine (version 2.0) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * November 15, 1997
 *
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

#include "dsp_defs.h"
#include "superlu_util.h"

int
dcopy_to_ucol(
	      int        jcol,	  /* in */
	      int        nseg,	  /* in */
	      int        *segrep,  /* in */
	      int        *repfnz,  /* in */
	      int        *perm_r,  /* in */
	      double     *dense,   /* modified - reset to zero on return */
	      GlobalLU_t *Glu      /* modified */
	      )
{
/* 
 * Gather from SPA dense[*] to global ucol[*].
 */
    int ksub, krep, ksupno;
    int i, k, kfnz, segsze;
    int fsupc, isub, irow;
    int jsupno, nextu;
    int new_next, mem_error;
    int       *xsup, *supno;
    int       *lsub, *xlsub;
    double    *ucol;
    int       *usub, *xusub;
    int       nzumax;

    double zero = 0.0;

    xsup    = Glu->xsup;
    supno   = Glu->supno;
    lsub    = Glu->lsub;
    xlsub   = Glu->xlsub;
    ucol    = Glu->ucol;
    usub    = Glu->usub;
    xusub   = Glu->xusub;
    nzumax  = Glu->nzumax;
    
    jsupno = supno[jcol];
    nextu  = xusub[jcol];
    k = nseg - 1;
    for (ksub = 0; ksub < nseg; ksub++) {
	krep = segrep[k--];
	ksupno = supno[krep];

	if ( ksupno != jsupno ) { /* Should go into ucol[] */
	    kfnz = repfnz[krep];
	    if ( kfnz != EMPTY ) {	/* Nonzero U-segment */

	    	fsupc = xsup[ksupno];
	        isub = xlsub[fsupc] + kfnz - fsupc;
	        segsze = krep - kfnz + 1;

		new_next = nextu + segsze;
		while ( new_next > nzumax ) {
		    if ((mem_error = dLUMemXpand(jcol, nextu, UCOL, &nzumax, Glu)) != 0)
			return (mem_error);
		    ucol = Glu->ucol;
		    if ((mem_error = dLUMemXpand(jcol, nextu, USUB, &nzumax, Glu)) != 0)
			return (mem_error);
		    usub = Glu->usub;
		    lsub = Glu->lsub;
		}
		
		for (i = 0; i < segsze; i++) {
		    irow = lsub[isub];
		    usub[nextu] = perm_r[irow];
		    ucol[nextu] = dense[irow];
		    dense[irow] = zero;
		    nextu++;
		    isub++;
		} 

	    }

	}

    } /* for each segment... */

    xusub[jcol + 1] = nextu;      /* Close U[*,jcol] */
    return 0;
}
