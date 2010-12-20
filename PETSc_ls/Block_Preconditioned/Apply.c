/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/




/* Include BJ Headers */
#include "BlockJacobiPcKsp.h"

/* Include solver calling sequence */

/***********************************************************************/
HYPRE_Int INCFACT_Apply ( void *solver_data, Vec b, Vec x)
     /* Wrapper around inc_fact_solve in form needed by Petsc */
{
  double    *INCFACT_b, *INCFACT_x;
  void      *incfact_data;
  Scalar     zero = 0.0;
  BJData    *BJ_data = solver_data;
  HYPRE_Int        ierr;
  HYPRE_Int        size;
  HYPRE_Int        ierr_incfactt, ierr_solver, ierr_input;
  HYPRE_Int        n, nnz, lenpmx;
  HYPRE_Int        scale, reorder;


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
 
