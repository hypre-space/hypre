/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * HYPRE_SStructVector interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"
#include "sstruct_mv.h"

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorcreate, HYPRE_SSTRUCTVECTORCREATE)
                                                              (int      *comm,
                                                               long int *grid,
                                                               long int *vector_ptr,
                                                               int      *ierr)
{
   *ierr = (int) (HYPRE_SStructVectorCreate( (MPI_Comm)             *comm,
                                             (HYPRE_SStructGrid)    *grid,
                                             (HYPRE_SStructVector *) vector_ptr ) );
}

/*--------------------------------------------------------------------------
HYPRE_SStructVectorDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectordestroy, HYPRE_SSTRUCTVECTORDESTROY)
                                                              (long int *vector,
                                                               int      *ierr)
{
   *ierr = (int) (HYPRE_SStructVectorDestroy( (HYPRE_SStructVector) *vector ) );
}

/*---------------------------------------------------------
HYPRE_SStructVectorInitialize
 * ----------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorinitialize, HYPRE_SSTRUCTVECTORINITIALIZE)
                                                              (long int *vector,
                                                               int      *ierr)
{
   *ierr = (int) (HYPRE_SStructVectorInitialize( (HYPRE_SStructVector) *vector ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorSetValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorsetvalues, HYPRE_SSTRUCTVECTORSETVALUES)
                                                              (long int *vector,
                                                               int      *part,
                                                               int      *index,
                                                               int      *var,
                                                               double   *value,
                                                               int      *ierr)
{
   *ierr = (int) (HYPRE_SStructVectorSetValues( (HYPRE_SStructVector) *vector,
                                                (int)                 *part,
                                                (int *)                index,
                                                (int)                 *var,
                                                (double *)             value ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorSetBoxValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorsetboxvalues, HYPRE_SSTRUCTVECTORSETBOXVALUES)
                                                              (long int *vector,
                                                               int      *part,
                                                               int      *ilower,
                                                               int      *iupper,
                                                               int      *var,
                                                               double   *values,
                                                               int      *ierr)
{
   *ierr = (int) (HYPRE_SStructVectorSetBoxValues( (HYPRE_SStructVector) *vector,
                                                   (int)                 *part,
                                                   (int *)                ilower,
                                                   (int *)                iupper,
                                                   (int)                 *var,
                                                   (double *)             values ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorAddToValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectoraddtovalues, HYPRE_SSTRUCTVECTORADDTOVALUES)
                                                              (long int *vector,
                                                               int      *part,
                                                               int      *index,
                                                               int      *var,
                                                               double   *value,
                                                               int      *ierr)
{
   *ierr = (int) (HYPRE_SStructVectorAddToValues( (HYPRE_SStructVector) *vector,
                                                  (int)                 *part,
                                                  (int *)                index,
                                                  (int)                 *var,
                                                  (double *)             value ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorAddToBoxValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectoraddtoboxvalu, HYPRE_SSTRUCTVECTORADDTOBOXVALU)
                                                              (long int *vector,
                                                               int      *part,
                                                               int      *ilower,
                                                               int      *iupper,
                                                               int      *var,
                                                               double   *values,
                                                               int      *ierr)
{
   *ierr = (int) (HYPRE_SStructVectorAddToBoxValues( (HYPRE_SStructVector) *vector,
                                                     (int)                 *part,
                                                     (int *)                ilower,
                                                     (int *)                iupper,
                                                     (int)                 *var,
                                                     (double *)             values ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorAssemble
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorassemble, HYPRE_SSTRUCTVECTORASSEMBLE)
                                                              (long int *vector,
                                                               int      *ierr)
{
   *ierr = (int) (HYPRE_SStructVectorAssemble( (HYPRE_SStructVector) *vector ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorGather
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorgather, HYPRE_SSTRUCTVECTORGATHER)
                                                              (long int *vector,
                                                               int      *ierr)
{
   *ierr = (int) (HYPRE_SStructVectorGather( (HYPRE_SStructVector) *vector ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorGetValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorgetvalues, HYPRE_SSTRUCTVECTORGETVALUES)
                                                              (long int *vector,
                                                               int      *part,
                                                               int      *index,
                                                               int      *var,
                                                               double   *value,
                                                               int      *ierr)
{
   *ierr = (int) (HYPRE_SStructVectorGetValues( (HYPRE_SStructVector) *vector,
                                                (int)                 *part,
                                                (int *)                index,
                                                (int)                 *var,
                                                (double *)             value ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorGetBoxValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorgetboxvalues, HYPRE_SSTRUCTVECTORGETBOXVALUES)
                                                              (long int *vector,
                                                               int      *part,
                                                               int      *ilower,
                                                               int      *iupper,
                                                               int      *var,
                                                               double   *values,
                                                               int      *ierr)
{
   *ierr = (int) (HYPRE_SStructVectorGetBoxValues( (HYPRE_SStructVector ) *vector,
                                                   (int)                  *part,
                                                   (int *)                 ilower,
                                                   (int *)                 iupper,
                                                   (int)                  *var,
                                                   (double *)              values ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorSetObjectType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorsetobjecttyp, HYPRE_SSTRUCTVECTORSETOBJECTTYP)
                                                              (long int *vector,
                                                               int      *type,
                                                               int      *ierr)
{
   *ierr = (int) (HYPRE_SStructVectorSetObjectType( (HYPRE_SStructVector) *vector,
                                                    (int)                 *type ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorGetObject
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorgetobject, HYPRE_SSTRUCTVECTORGETOBJECT)
                                                              (long int *vector,
                                                               void    **object,
                                                               int      *ierr)
{
   *ierr = (int) (HYPRE_SStructVectorGetObject( (HYPRE_SStructVector) *vector,
                                                (void *)              *object ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorPrint
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorprint, HYPRE_SSTRUCTVECTORPRINT)
                                                              (const char *filename,
                                                               long int   *vector,
                                                               int        *all,
                                                               int        *ierr)
{
   *ierr = (int) (HYPRE_SStructVectorPrint( (const char * )        filename,
                                            (HYPRE_SStructVector) *vector,
                                            (int)                 *all ) );
}
