/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision 1.0 $
 *********************************************************************EHEADER*/

#include <mpi.h>
#include <HYPRE_config.h>
#include "fortran.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mpi_comm_f2c, HYPRE_MPI_COMM_F2C)(int *c_comm,
                                                        int *f_comm,
                                                        int *ierr)
{
#ifdef HYPRE_HAVE_MPI_COMM_F2C

   *c_comm = (int)MPI_Comm_f2c(*f_comm);

   if (sizeof(MPI_Comm) > sizeof(int))
      *ierr = 1;
   else
      *ierr = 0;

#else

   *c_comm = *f_comm;
   *ierr = 0;

#endif
}
