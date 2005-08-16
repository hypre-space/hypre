/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * HYPRE_ParCSRint Fortran interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"
#include "HYPRE_parcsr_int.h"

int hypre_ParVectorSize( void *x );
int aux_maskCount( int n, int *mask );
void aux_indexFromMask( int n, int *mask, int *index );


/*--------------------------------------------------------------------------
 * hypre_ParSetRandomValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parsetrandomvalues, HYPRE_PARSETRANDOMVALUES)
               (long int *v, int *seed, int *ierr)
{
   *ierr = (int) ( HYPRE_ParVectorSetRandomValues( (HYPRE_ParVector) *v,
                                                   (int)             *seed));
}

/*--------------------------------------------------------------------------
 * hypre_ParPrintVector
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parprintvector, HYPRE_PARPRINTVECTOR)
               (long int *v, char *file, int *ierr)
{
   *ierr = (int) ( hypre_ParVectorPrint( (hypre_ParVector *) v,
                                         (char *)            file));
}

/*--------------------------------------------------------------------------
 * hypre_ParReadVector
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parreadvector, HYPRE_PARREADVECTOR)
               (int *comm, char *file, int *ierr)
{
   *ierr = 0;

   (void*) (hypre_ParReadVector( (MPI_Comm)    *comm, 
                                (const char *) file ));
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorSize
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parvectorsize, HYPRE_PARVECTORSIZE)
               (long int *x, int *ierr)
{
   *ierr = (int) ( hypre_ParVectorSize( (void *) x) );
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMultiVectorPrint
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrmultivectorprint, HYPRE_PARCSRMULTIVECTORPRINT)
               (long int *x, char *file, int *ierr)
{
   *ierr = (int) ( hypre_ParCSRMultiVectorPrint( (void *)       x, 
                                                 (const char *) file));
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMultiVectorRead
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrmultivectorread, HYPRE_PARCSRMULTIVECTORREAD)
               (int *comm, long int *ii, char *file, int *ierr)
{
   *ierr = 0;

   (void *) hypre_ParCSRMultiVectorRead( (MPI_Comm)    *comm,
                                         (void *)       ii, 
                                         (const char *) file );
}

/*--------------------------------------------------------------------------
 * aux_maskCount
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(aux_maskcount, AUX_MASKCOUNT)
               (int *n, int *mask, int *ierr)
{
   *ierr = (int) ( aux_maskCount( (int)   *n,
                                  (int *)  mask ));
}

/*--------------------------------------------------------------------------
 * aux_indexFromMask
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(aux_indexfrommask, AUX_INDEXFROMMASK)
               (int *n, int *mask, int *index, int *ierr)
{
   *ierr = 0;

  (void) ( aux_indexFromMask( (int)   *n, 
                              (int *)  mask,
                              (int *)  index ));
}

/*--------------------------------------------------------------------------
 * HYPRE_TempParCSRSetupInterpreter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_tempparcsrsetupinterprete, HYPRE_TEMPPARCSRSETUPINTERPRETE)
               (long int *i, int *ierr)
{
   *ierr = (int) ( HYPRE_TempParCSRSetupInterpreter( (mv_InterfaceInterpreter *) i ));
}

/*--------------------------------------------------------------------------
 * HYPRE_TempParCSRSetupInterpreter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrsetupinterpreter, HYPRE_PARCSRSETUPINTERPRETER)
               (long int *i, int *ierr)
{
   *ierr = (int) ( HYPRE_ParCSRSetupInterpreter( (mv_InterfaceInterpreter *) i ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRSetupMatvec
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrsetupmatvec, HYPRE_PARCSRSETUPMATVEC)
               (long int *mv, int *ierr)
{
   *ierr = (int) ( HYPRE_ParCSRSetupMatvec( (HYPRE_MatvecFunctions *) mv));
}
