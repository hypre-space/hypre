/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

/**************************************************************************
 **************************************************************************
 * MLI_FEData utilities functions
 **************************************************************************
 **************************************************************************/

#include <iostream.h>
#include <stdio.h>
#include <string.h>
#include "HYPRE.h"
#include <mpi.h>
#include "parcsr_mv/parcsr_mv.h"
#include "mli_fedata.h"

/**************************************************************************
 * functions 
 *-----------------------------------------------------------------------*/

void MLI_FEData_GetParCSRelement_node(MPI_Comm,MLI_FEData&,HYPRE_ParCSRMatrix*);
void MLI_FEData_GetParCSRelement_face(MPI_Comm,MLI_FEData&,HYPRE_ParCSRMatrix*);
void MLI_FEData_GetParCSRface_node(   MPI_Comm,MLI_FEData&,HYPRE_ParCSRMatrix*);
void MLI_FEData_GetParCSRnode_element(MPI_Comm,MLI_FEData&,HYPRE_ParCSRMatrix*);
void MLI_FEData_GetParCSRface_element(MPI_Comm,MLI_FEData&,HYPRE_ParCSRMatrix*);
void MLI_FEData_GetParCSRnode_face   (MPI_Comm,MLI_FEData&,HYPRE_ParCSRMatrix*);

