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
 * HYPRE_NewStructVector
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_newstructvector)( int      *comm,
                                        long int *grid,
                                        long int *stencil,
                                        long int *vector,
                                        int      *ierr    )
{
   *ierr = (int) ( HYPRE_NewStructVector( (MPI_Comm)             *comm,
                                          (HYPRE_StructGrid)     *grid,
                                          (HYPRE_StructStencil)  *stencil,
                                          (HYPRE_StructVector *) vector   ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FreeStructVector
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_freestructvector)( long int *vector,
                                         int      *ierr   )
{
   *ierr = (int) ( HYPRE_FreeStructVector( (HYPRE_StructVector) *vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_InitializeStructVector
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_initializestructvector)( long int *vector,
                                               int      *ierr   )
{
   *ierr = (int) ( HYPRE_InitializeStructVector( (HYPRE_StructVector) *vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetStructVectorValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_setstructvectorvalues)( long int *vector,
                                              int      *grid_index,
                                              double   *values,
                                              int      *ierr       )
{
   *ierr = (int) ( HYPRE_SetStructVectorValues( (HYPRE_StructVector) *vector,
                                                (int *)              grid_index,
                                                (double)             *values     ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GetStructVectorValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_getstructvectorvalues)( long int *vector,
                                              int      *grid_index,
                                              double   *values_ptr,
                                              int      *ierr       )
{
   *ierr = (int) ( HYPRE_GetStructVectorValues( (HYPRE_StructVector) *vector,
                                                (int *)              grid_index,
                                                (double *)           values_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetStructVectorBoxValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_setstructvectorboxvalues)( long int *vector,
                                                 int      *ilower,
                                                 int      *iupper,
                                                 double   *values,
                                                 int      *ierr   )
{
   *ierr = (int) ( HYPRE_SetStructVectorBoxValues( (HYPRE_StructVector) *vector,
                                                   (int *)              ilower,
                                                   (int *)              iupper,
                                                   (double *)           values  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GetStructVectorBoxValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_getstructvectorboxvalues)( long int *vector,
                                                 int      *ilower,
                                                 int      *iupper,
                                                 double   *values,
                                                 int      *ierr   )
{
   *ierr = (int) ( HYPRE_GetStructVectorBoxValues( (HYPRE_StructVector) *vector,
                                                   (int *)              ilower,
                                                   (int *)              iupper,
                                                   (double *)           values  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AssembleStructVector
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_assemblestructvector)( long int *vector,
                                             int      *ierr   )
{
   *ierr = (int) ( HYPRE_AssembleStructVector( (HYPRE_StructVector) *vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetStructVectorNumGhost
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_setstructvectornumghost)( long int *vector,
                                                int      *num_ghost,
                                                int      *ierr      )
{
   *ierr = (int) ( HYPRE_SetStructVectorNumGhost( (HYPRE_StructVector) *vector,
                                                  (int *)              num_ghost ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetStructVectorConstantValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_setstructvectorconstantva)( long int *vector,
                                                  double   *values,
                                                  int      *ierr   )
{
   *ierr = (int) ( HYPRE_SetStructVectorConstantValues( (HYPRE_StructVector)  *vector,
                                                        (double)              *values ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GetMigrateStructVectorCommPkg
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_getmigratestructvectorcom)( long int *from_vector,
                                                  long int *to_vector,
                                                  long int *comm_pkg,
                                                  int      *ierr        )
{
   *ierr = (int) ( HYPRE_GetMigrateStructVectorCommPkg( (HYPRE_StructVector) *from_vector,
                                                        (HYPRE_StructVector) *to_vector,
                                                        (HYPRE_CommPkg *)    comm_pkg    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MigrateStructVector
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_migratestructvector)( long int *comm_pkg,
                                            long int *from_vector,
                                            long int *to_vector,
                                            int      *ierr        )
{
   *ierr = (int) ( HYPRE_MigrateStructVector( (HYPRE_CommPkg)      *comm_pkg,
                                              (HYPRE_StructVector) *from_vector,
                                              (HYPRE_StructVector) *to_vector   ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FreeCommPkg
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_freecommpkg)( long int *comm_pkg,
                                    int      *ierr     )
{
   *ierr = (int) ( HYPRE_FreeCommPkg( (HYPRE_CommPkg) *comm_pkg ) );
}
