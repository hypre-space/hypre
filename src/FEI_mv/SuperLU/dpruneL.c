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
 * $Revision: 1.4 $
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

void
dpruneL(
       const int  jcol,	     /* in */
       const int  *perm_r,   /* in */
       const int  pivrow,    /* in */
       const int  nseg,	     /* in */
       const int  *segrep,   /* in */
       const int  *repfnz,   /* in */
       int        *xprune,   /* out */
       GlobalLU_t *Glu       /* modified - global LU data structures */
       )
{
/*
 * Purpose
 * =======
 *   Prunes the L-structure of supernodes whose L-structure
 *   contains the current pivot row "pivrow"
 *
 */
    double     utemp;
    int        jsupno, irep, irep1, kmin, kmax, krow, movnum;
    int        i, ktemp, minloc, maxloc;
    int        do_prune; /* logical variable */
    int        *xsup, *supno;
    int        *lsub, *xlsub;
    double     *lusup;
    int        *xlusup;

    xsup       = Glu->xsup;
    supno      = Glu->supno;
    lsub       = Glu->lsub;
    xlsub      = Glu->xlsub;
    lusup      = Glu->lusup;
    xlusup     = Glu->xlusup;
    
    /*
     * For each supernode-rep irep in U[*,j]
     */
    jsupno = supno[jcol];
    for (i = 0; i < nseg; i++) {

	irep = segrep[i];
	irep1 = irep + 1;
	do_prune = FALSE;

	/* Don't prune with a zero U-segment */
 	if ( repfnz[irep] == EMPTY )
		continue;

     	/* If a snode overlaps with the next panel, then the U-segment 
   	 * is fragmented into two parts -- irep and irep1. We should let
	 * pruning occur at the rep-column in irep1's snode. 
	 */
	if ( supno[irep] == supno[irep1] ) 	/* Don't prune */
		continue;

	/*
	 * If it has not been pruned & it has a nonz in row L[pivrow,i]
	 */
	if ( supno[irep] != jsupno ) {
	    if ( xprune[irep] >= xlsub[irep1] ) {
		kmin = xlsub[irep];
		kmax = xlsub[irep1] - 1;
		for (krow = kmin; krow <= kmax; krow++) 
		    if ( lsub[krow] == pivrow ) {
			do_prune = TRUE;
			break;
		    }
	    }
	    
    	    if ( do_prune ) {

	     	/* Do a quicksort-type partition
	     	 * movnum=TRUE means that the num values have to be exchanged.
	     	 */
	        movnum = FALSE;
	        if ( irep == xsup[supno[irep]] ) /* Snode of size 1 */
			movnum = TRUE;

	        while ( kmin <= kmax ) {

	    	    if ( perm_r[lsub[kmax]] == EMPTY ) 
			kmax--;
		    else if ( perm_r[lsub[kmin]] != EMPTY )
			kmin++;
		    else { /* kmin below pivrow, and kmax above pivrow: 
		            * 	interchange the two subscripts
			    */
		        ktemp = lsub[kmin];
		        lsub[kmin] = lsub[kmax];
		        lsub[kmax] = ktemp;

			/* If the supernode has only one column, then we
 			 * only keep one set of subscripts. For any subscript 
			 * interchange performed, similar interchange must be 
			 * done on the numerical values.
 			 */
		        if ( movnum ) {
		    	    minloc = xlusup[irep] + (kmin - xlsub[irep]);
		    	    maxloc = xlusup[irep] + (kmax - xlsub[irep]);
			    utemp = lusup[minloc];
		  	    lusup[minloc] = lusup[maxloc];
			    lusup[maxloc] = utemp;
		        }

		        kmin++;
		        kmax--;

		    }

	        } /* while */

	        xprune[irep] = kmin;	/* Pruning */

#ifdef CHK_PRUNE
	printf("    After dpruneL(),using col %d:  xprune[%d] = %d\n", 
			jcol, irep, kmin);
#endif
	    } /* if do_prune */

	} /* if */

    } /* for each U-segment... */
}
