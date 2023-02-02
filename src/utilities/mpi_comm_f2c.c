/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <HYPRE_config.h>
#include "fortran.h"
#ifndef HYPRE_SEQUENTIAL
#include <mpi.h>
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

#if 0 /* This function is problematic and no longer needed anyway. */
void
hypre_F90_IFACE(hypre_mpi_comm_f2c, HYPRE_MPI_COMM_F2C)
(hypre_F90_Obj  *c_comm,
 hypre_F90_Comm *f_comm,
 hypre_F90_Int  *ierr)
{
   *c_comm = (hypre_F90_Obj) hypre_MPI_Comm_f2c( (hypre_int) * f_comm );
   *ierr = 0;
}
#endif
