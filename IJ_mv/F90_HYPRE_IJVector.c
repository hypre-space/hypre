/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * HYPRE_IJMatrix Fortran interface
 *
 *****************************************************************************/

#include "./IJ_mv.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijvectorcreate, HYPRE_IJVECTORCREATE)(
                                                    int      *comm,
                                                    int      *jlower,
                                                    int      *jupper,
                                                    long int *vector,
                                                    int      *ierr    )
{
   *ierr = (int) ( HYPRE_IJVectorCreate( (MPI_Comm)         *comm,
                                         (int)              *jlower,
                                         (int)              *jupper, 
                                         (HYPRE_IJVector *)  vector  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijvectordestroy, HYPRE_IJVECTORDESTROY)(
                                                    long int *vector,
                                                    int      *ierr    )
{
   *ierr = (int) ( HYPRE_IJVectorDestroy( (HYPRE_IJVector) *vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorInitialize
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijvectorinitialize, HYPRE_IJVECTORINITIALIZE)(
                                                    long int *vector,
                                                    int      *ierr    )
{
   *ierr = (int) ( HYPRE_IJVectorInitialize( (HYPRE_IJVector) *vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorSetValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijvectorsetvalues, HYPRE_IJVECTORSETVALUES)(
                                                    long int *vector,
                                                    int      *num_values,
                                                    int      *indices,
                                                    double   *values,
                                                    int      *ierr        )
{
   *ierr = (int) ( HYPRE_IJVectorSetValues( (HYPRE_IJVector) *vector,
                                            (int)            *num_values,
                                            (const int *)     indices,
                                            (const double *)  values      ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorAddToValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijvectoraddtovalues, HYPRE_IJVECTORADDTOVALUES)(
                                                    long int *vector,
                                                    int      *num_values,
                                                    int      *indices,
                                                    double   *values,
                                                    int      *ierr        )
{
   *ierr = (int) ( HYPRE_IJVectorAddToValues( (HYPRE_IJVector) *vector,
                                              (int)            *num_values,
                                              (const int *)     indices,
                                              (const double *)  values      ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorAssemble
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijvectorassemble, HYPRE_IJVECTORASSEMBLE)(
                                                    long int *vector,
                                                    int      *ierr    )
{
   *ierr = (int) ( HYPRE_IJVectorAssemble( (HYPRE_IJVector) *vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorGetValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijvectorgetvalues, HYPRE_IJVECTORGETVALUES)(
                                                    long int  *vector,
                                                    const int *num_values,
                                                    const int *indices,
                                                    double   *values,
                                                    int      *ierr        )
{
   *ierr = (int) ( HYPRE_IJVectorGetValues( (HYPRE_IJVector) *vector,
                                            (int)            *num_values,
                                            (const int *)     indices,
                                            (double *)        values      ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorSetObjectType
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijvectorsetobjecttype, HYPRE_IJVECTORSETOBJECTTYPE)(
                                                    long int  *vector,
                                                    const int *type,
                                                    int       *ierr    )
{
   *ierr = (int) ( HYPRE_IJVectorSetObjectType( (HYPRE_IJVector) *vector,
                                                (int)            *type    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorGetObjectType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijvectorgetobjecttype, HYPRE_IJVECTORGETOBJECTTYPE)(
                                                    long int *vector,
                                                    int      *type,
                                                    int      *ierr    )
{
   *ierr = (int) (HYPRE_IJVectorGetObjectType( (HYPRE_IJVector) *vector,
                                               (int *)           type    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorGetObject
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijvectorgetobject, HYPRE_IJVECTORGETOBJECT)(
                                                    long int *vector,
                                                    long int *object,
                                                    int      *ierr    )
{
   *ierr = (long int) (HYPRE_IJVectorGetObject( (HYPRE_IJVector) *vector,
                                                (void **)         object  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorRead
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijvectorread, HYPRE_IJVECTORREAD)( char     *filename,
                                                         long int *comm,
                                                         int      *object_type,
                                                         long int *vector,
                                                         int      *ierr      )
{
   *ierr = (int) (HYPRE_IJVectorRead( (char *)            filename,
                                      (MPI_Comm)         *comm,
                                      (int)              *object_type,
                                      (HYPRE_IJVector *)  vector       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorPrint
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijvectorprint, HYPRE_IJVECTORPRINT)( long int *vector,
                                                           char     *filename,
                                                           int      *ierr      )
{
   *ierr = (int) (HYPRE_IJVectorPrint( (HYPRE_IJVector) *vector,
                                       (char *)          filename ) );
}
