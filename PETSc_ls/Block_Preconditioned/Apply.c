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



/* Include BJ Headers */
#include "BlockJacobiPcKsp.h"

/* Include solver calling sequence */

/***********************************************************************/
int INCFACT_Apply ( void *solver_data, Vec b, Vec x)
     /* Wrapper around inc_fact_solve in form needed by Petsc */
{
  double    *INCFACT_b, *INCFACT_x;
  void      *incfact_data;
  Scalar     zero = 0.0;
  BJData    *BJ_data = solver_data;
  int        ierr;
  int        size;
  int        ierr_incfactt, ierr_solver, ierr_input;
  int        n, nnz, lenpmx;
  int        scale, reorder;


  /***********                                    ***********/
  /* Convert Petsc formatted vectors to that expected for INCFACT */
  ierr = VecGetArray( b, &INCFACT_b ); CHKERRA(ierr);
  ierr = VecGetLocalSize( b, &size ); CHKERRA(ierr);

  ierr = VecSet( &zero, x );
  ierr = VecGetArray( x, &INCFACT_x ); CHKERRA(ierr);
  ierr = VecGetLocalSize( x, &size ); CHKERRA(ierr);
  /***********                                    ***********/

  incfact_data = BJDataLsData( BJ_data );

  ierr = incfact_solve ( incfact_data, INCFACT_x, INCFACT_b );
  /*
  ierr = VecCopy( b, x );
  */

  ierr = VecRestoreArray( b, &INCFACT_b); CHKERRA(ierr);
  ierr = VecRestoreArray( x, &INCFACT_x); CHKERRA(ierr);

  return(ierr); 

}
 
