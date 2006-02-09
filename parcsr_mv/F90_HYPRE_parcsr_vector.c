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
 * HYPRE_ParVector Fortran interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parvectorcreate, HYPRE_PARVECTORCREATE)( int      *comm,
                                     int      *global_size,
                                     long int *partitioning,
                                     long int *vector,
                                     int      *ierr          )
{
   *ierr = (long int)
             ( HYPRE_ParVectorCreate( (MPI_Comm) *comm,
                                      (int)      *global_size,
                                      (int *)    *partitioning,
                                      (HYPRE_ParVector *)  vector ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_ParMultiVectorCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parmultivectorcreate, HYPRE_PARMULTIVECTORCREATE)
                                   ( int      *comm,
                                     int      *global_size,
                                     long int *partitioning,
                                     int      *number_vectors,
                                     long int *vector,
                                     int      *ierr          )
{
   *ierr = (long int)
             ( HYPRE_ParMultiVectorCreate( (MPI_Comm) *comm,
                                      (int)      *global_size,
                                      (int *)    *partitioning,
                                      (int)      *number_vectors,
                                      (HYPRE_ParVector *)  vector ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parvectordestroy, HYPRE_PARVECTORDESTROY)( long int *vector,
                                         int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParVectorDestroy( (HYPRE_ParVector) *vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorInitialize
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parvectorinitialize, HYPRE_PARVECTORINITIALIZE)( long int *vector,
                                            int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParVectorInitialize( (HYPRE_ParVector) *vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorRead
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parvectorread, HYPRE_PARVECTORREAD)( int      *comm,
                                      long int *vector,
                                      char     *file_name,
                                      int      *ierr       )
{
   *ierr = (int) ( HYPRE_ParVectorRead( (MPI_Comm) *comm,
                                        (char *)    file_name,
                                        (HYPRE_ParVector *) vector ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorPrint
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parvectorprint, HYPRE_PARVECTORPRINT)( long int *vector,
                                       char     *fort_file_name,
                                       int      *fort_file_name_size,
                                       int      *ierr       )
{
   int i;
   char *c_file_name;

   c_file_name = hypre_CTAlloc(char, *fort_file_name_size);

   for (i = 0; i < *fort_file_name_size; i++)
     c_file_name[i] = fort_file_name[i];

   *ierr = (int) ( HYPRE_ParVectorPrint ( (HYPRE_ParVector) *vector,
                                          (char *)           c_file_name ) );

   hypre_TFree(c_file_name);

}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorSetConstantValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parvectorsetconstantvalue, HYPRE_PARVECTORSETCONSTANTVALUE)
                                          ( long int *vector,
                                            double   *value,
                                            int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParVectorSetConstantValues( (HYPRE_ParVector) *vector,
                                                     (double)          *value) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorSetRandomValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parvectorsetrandomvalues, HYPRE_PARVECTORSETRANDOMVALUES)
                                          ( long int *vector,
                                            int      *seed,
                                            int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParVectorSetRandomValues( (HYPRE_ParVector) *vector,
                                                   (int)             *seed) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorCopy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parvectorcopy, HYPRE_PARVECTORCOPY)
                                          ( long int *x,
                                            long int *y,
                                            int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParVectorCopy( (HYPRE_ParVector) *x,
                                        (HYPRE_ParVector) *y) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorCloneShallow
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parvectorcloneshallow, HYPRE_PARVECTORCLONESHALLOW)
                                          ( long int *x,
                                            int      *ierr    )
{
   *ierr = (long int) ( HYPRE_ParVectorCloneShallow( (HYPRE_ParVector) *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorScale
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parvectorscale, HYPRE_PARVECTORSCALE)
                                          ( double   *value,
                                            long int *x,
                                            int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParVectorScale( (double)          *value,
                                         (HYPRE_ParVector) *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorAxpy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parvectoraxpy, HYPRE_PARVECTORAXPY)
                                          ( double   *value,
                                            long int *x,
                                            long int *y,
                                            int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParVectorAxpy( (double)          *value,
                                        (HYPRE_ParVector) *x,
                                        (HYPRE_ParVector) *y ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorInnerProd
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parvectorinnerprod, HYPRE_PARVECTORINNERPROD)
                                           (long int *x,
                                            long int *y,
                                            double   *prod,
                                            int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParVectorInnerProd( (HYPRE_ParVector) *x,
                                             (HYPRE_ParVector) *y,
                                             (double *)         prod ) );
}
