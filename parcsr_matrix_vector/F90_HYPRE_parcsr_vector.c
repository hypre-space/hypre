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
