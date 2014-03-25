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
HYPRE_Int ILU_Apply ( void *solver_data, Vec b, Vec x)
     /* Wrapper around inc_fact_solve in form needed by Petsc */
{
  HYPRE_Real    *ILU_b, *ILU_x;
  void      *ilu_data;
  Scalar     zero = 0.0;
  BJData    *BJ_data = solver_data;
  HYPRE_Int        ierr;
  HYPRE_Int        size;
  HYPRE_Int        ierr_ilut, ierr_solver, ierr_input;
  HYPRE_Int        n, nnz, lenpmx;
  HYPRE_Int        scale, reorder;


  /***********                                    ***********/
  /* Convert Petsc formatted vectors to that expected for ILU */
  ierr = VecGetArray( b, &ILU_b ); CHKERRA(ierr);
  ierr = VecGetLocalSize( b, &size ); CHKERRA(ierr);

  ierr = VecSet( &zero, x );
  ierr = VecGetArray( x, &ILU_x ); CHKERRA(ierr);
  ierr = VecGetLocalSize( x, &size ); CHKERRA(ierr);
  /***********                                    ***********/

  ilu_data = BJDataLsData( BJ_data );

  ierr = ilu_solve ( ilu_data, ILU_x, ILU_b );
  /*
  ierr = VecCopy( b, x );
  */

  ierr = VecRestoreArray( b, &ILU_b); CHKERRA(ierr);
  ierr = VecRestoreArray( x, &ILU_x); CHKERRA(ierr);

  return(ierr); 

}
 
