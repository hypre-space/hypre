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
hypre_F90_IFACE(hypre_createstructvector)( int      *comm,
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
hypre_F90_IFACE(hypre_destroystructvector)( long int *vector,
                                            int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructVectorDestroy( (HYPRE_StructVector) *vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorInitialize
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_initializestructvector)( long int *vector,
                                               int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_StructVectorInitialize( (HYPRE_StructVector) *vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorSetValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_setstructvectorvalues)( long int *vector,
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
 * HYPRE_StructVectorGetValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_getstructvectorvalues)( long int *vector,
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
 * HYPRE_StructVectorSetBoxValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_setstructvectorboxvalues)( long int *vector,
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
 * HYPRE_StructVectorGetBoxValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_getstructvectorboxvalues)( long int *vector,
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
hypre_F90_IFACE(hypre_assemblestructvector)( long int *vector,
                                             int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_StructVectorAssemble( (HYPRE_StructVector) *vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorSetNumGhost
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_setstructvectornumghost)( long int *vector,
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
hypre_F90_IFACE(hypre_setstructvectorconstantva)( long int *vector,
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
hypre_F90_IFACE(hypre_getmigratestructvectorcom)( long int *from_vector,
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
hypre_F90_IFACE(hypre_migratestructvector)( long int *comm_pkg,
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
