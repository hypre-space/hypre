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
 * HYPRE_StructVector interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structvectorcreate)( int      *comm,
                                           long int *grid,
                                           long int *stencil,
                                           long int *vector,
                                           int      *ierr    )
{
   *ierr = (int)
      ( HYPRE_StructVectorCreate( (MPI_Comm)             *comm,
                                  (HYPRE_StructGrid)     *grid,
                                  (HYPRE_StructStencil)  *stencil,
                                  (HYPRE_StructVector *) vector   ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structvectordestroy)( long int *vector,
                                            int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructVectorDestroy( (HYPRE_StructVector) *vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorInitialize
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structvectorinitialize)( long int *vector,
                                               int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_StructVectorInitialize( (HYPRE_StructVector) *vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorSetValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structvectorsetvalues)( long int *vector,
                                              int      *grid_index,
                                              double   *values,
                                              int      *ierr       )
{
   *ierr = (int)
      ( HYPRE_StructVectorSetValues( (HYPRE_StructVector) *vector,
                                     (int *)              grid_index,
                                     (double)             *values     ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorSetBoxValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structvectorsetboxvalues)( long int *vector,
                                                 int      *ilower,
                                                 int      *iupper,
                                                 double   *values,
                                                 int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_StructVectorSetBoxValues( (HYPRE_StructVector) *vector,
                                        (int *)              ilower,
                                        (int *)              iupper,
                                        (double *)           values  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorAddToValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structvectoraddtovalues)( long int *vector,
                                                int      *grid_index,
                                                double   *values,
                                                int      *ierr       )
{
   *ierr = (int)
      ( HYPRE_StructVectorAddToValues( (HYPRE_StructVector) *vector,
                                       (int *)              grid_index,
                                       (double)             *values     ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorAddToBoxValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structvectoraddtoboxvalues)( long int *vector,
                                                   int      *ilower,
                                                   int      *iupper,
                                                   double   *values,
                                                   int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_StructVectorAddToBoxValues( (HYPRE_StructVector) *vector,
                                          (int *)              ilower,
                                          (int *)              iupper,
                                          (double *)           values  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorGetValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structvectorgetvalues)( long int *vector,
                                              int      *grid_index,
                                              double   *values_ptr,
                                              int      *ierr       )
{
   *ierr = (int)
      ( HYPRE_StructVectorGetValues( (HYPRE_StructVector) *vector,
                                     (int *)              grid_index,
                                     (double *)           values_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorGetBoxValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structvectorgetboxvalues)( long int *vector,
                                                 int      *ilower,
                                                 int      *iupper,
                                                 double   *values,
                                                 int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_StructVectorGetBoxValues( (HYPRE_StructVector) *vector,
                                        (int *)              ilower,
                                        (int *)              iupper,
                                        (double *)           values  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorAssemble
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structvectorassemble)( long int *vector,
                                             int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_StructVectorAssemble( (HYPRE_StructVector) *vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorSetNumGhost
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_structvectorsetnumghost)( long int *vector,
                                                int      *num_ghost,
                                                int      *ierr      )
{
   *ierr = (int)
      ( HYPRE_StructVectorSetNumGhost( (HYPRE_StructVector) *vector,
                                       (int *)              num_ghost ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorSetConstantValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structvectorsetconstantva)( long int *vector,
                                                  double   *values,
                                                  int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_StructVectorSetConstantValues( (HYPRE_StructVector)  *vector,
                                             (double)              *values ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorGetMigrateCommPkg
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structvectorgetmigratecom)( long int *from_vector,
                                                  long int *to_vector,
                                                  long int *comm_pkg,
                                                  int      *ierr        )
{
   *ierr = (int)
      ( HYPRE_StructVectorGetMigrateCommPkg( (HYPRE_StructVector) *from_vector,
                                             (HYPRE_StructVector) *to_vector,
                                             (HYPRE_CommPkg *)    comm_pkg    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorMigrate
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structvectormigrate)( long int *comm_pkg,
                                            long int *from_vector,
                                            long int *to_vector,
                                            int      *ierr        )
{
   *ierr = (int)
      ( HYPRE_StructVectorMigrate( (HYPRE_CommPkg)      *comm_pkg,
                                   (HYPRE_StructVector) *from_vector,
                                   (HYPRE_StructVector) *to_vector   ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CommPkgDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_destroycommpkg)( long int *comm_pkg,
                                       int      *ierr     )
{
   *ierr = (int)
      ( HYPRE_CommPkgDestroy( (HYPRE_CommPkg) *comm_pkg ) );
}
