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




/* Include headers for problem and solver data structure */
#include "BlockJacobiPcKsp.h"


HYPRE_Int BlockJacobiICPcKspSolve( void *data, Mat A, Vec x, Vec b )
     /* Uses IC as an approximate linear system solver on each
        processor as the block solver in BlockJacobi Preconditioner */
{


  BJData   *bj_data;
  SLES     *sles_ptr;  
  HYPRE_Int       flg, its;


  bj_data = (BJData *) data;

  sles_ptr = BJDataSles_ptr( bj_data );

  /* Call Petsc solver */
#if 0
  hypre_printf("about to call slessolve\n");
#endif

  flg = SLESSolve(*sles_ptr,b,x,&its); CHKERRA(flg);

#if 0
  PetscPrintf(hypre_MPI_COMM_WORLD, "iterations = %d\n",its);
#endif


  return flg;
}

