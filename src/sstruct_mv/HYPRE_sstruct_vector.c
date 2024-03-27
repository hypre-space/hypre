/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_SStructVector interface
 *
 *****************************************************************************/

#include "_hypre_sstruct_mv.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructVectorCreate( MPI_Comm              comm,
                           HYPRE_SStructGrid     grid,
                           HYPRE_SStructVector  *vector_ptr )
{
   hypre_SStructVector   *vector;
   HYPRE_Int              nparts;
   hypre_SStructPVector **pvectors;
   MPI_Comm               pcomm;
   hypre_SStructPGrid    *pgrid;
   HYPRE_Int              part;

   vector = hypre_TAlloc(hypre_SStructVector, 1, HYPRE_MEMORY_HOST);

   hypre_SStructVectorComm(vector) = comm;
   hypre_SStructVectorNDim(vector) = hypre_SStructGridNDim(grid);
   hypre_SStructGridRef(grid, &hypre_SStructVectorGrid(vector));
   hypre_SStructVectorObjectType(vector) = HYPRE_SSTRUCT;
   nparts = hypre_SStructGridNParts(grid);
   hypre_SStructVectorNParts(vector) = nparts;
   pvectors = hypre_TAlloc(hypre_SStructPVector *, nparts, HYPRE_MEMORY_HOST);
   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGridPGrid(grid, part);
      pcomm = hypre_SStructPGridComm(pgrid);
      hypre_SStructPVectorCreate(pcomm, pgrid, &pvectors[part]);
   }
   hypre_SStructVectorPVectors(vector)   = pvectors;
   hypre_SStructVectorIJVector(vector)   = NULL;

   /* GEC1002 initializing to NULL */

   hypre_SStructVectorDataIndices(vector) = NULL;
   hypre_SStructVectorData(vector)        = NULL;

   /* GEC1002 moving the creation of the ijvector the the initialize part
    *   ilower = hypre_SStructGridStartRank(grid);
    *   iupper = ilower + hypre_SStructGridLocalSize(grid) - 1;
    *  HYPRE_IJVectorCreate(comm, ilowergh, iuppergh,
    *                  &hypre_SStructVectorIJVector(vector)); */

   hypre_SStructVectorIJVector(vector)   = NULL;
   hypre_SStructVectorParVector(vector)  = NULL;
   hypre_SStructVectorGlobalSize(vector) = 0;
   hypre_SStructVectorRefCount(vector)   = 1;
   hypre_SStructVectorDataSize(vector)   = 0;
   hypre_SStructVectorObjectType(vector) = HYPRE_SSTRUCT;

   *vector_ptr = vector;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructVectorDestroy( HYPRE_SStructVector vector )
{
   HYPRE_Int              nparts;
   hypre_SStructPVector **pvectors;
   HYPRE_Int              part;
   HYPRE_Int              vector_type;
   HYPRE_MemoryLocation   memory_location;

   /* GEC1002 destroying data indices and data in vector  */
   if (vector)
   {
      memory_location = hypre_SStructVectorMemoryLocation(vector);

      vector_type = hypre_SStructVectorObjectType(vector);
      hypre_SStructVectorRefCount(vector) --;
      if (hypre_SStructVectorRefCount(vector) == 0)
      {
         HYPRE_SStructGridDestroy(hypre_SStructVectorGrid(vector));
         nparts   = hypre_SStructVectorNParts(vector);
         pvectors = hypre_SStructVectorPVectors(vector);
         for (part = 0; part < nparts; part++)
         {
            hypre_SStructPVectorDestroy(pvectors[part]);
         }
         hypre_TFree(pvectors, HYPRE_MEMORY_HOST);
         HYPRE_IJVectorDestroy(hypre_SStructVectorIJVector(vector));

         /* GEC1002 the ijdestroy takes care of the data when the
          * vector is type HYPRE_SSTRUCT. This is a result that the
          * ijvector does not use the owndata flag in the data structure
          * unlike the struct vector                               */

         /* GEC if data has been allocated then free the pointer */
         hypre_TFree(hypre_SStructVectorDataIndices(vector), HYPRE_MEMORY_HOST);

         if (hypre_SStructVectorData(vector) && (vector_type == HYPRE_PARCSR))
         {
            hypre_TFree(hypre_SStructVectorData(vector), memory_location);
         }

         hypre_TFree(vector, HYPRE_MEMORY_HOST);
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * GEC1002 changes to initialize the vector with a data chunk
 * that includes all the part,var pieces instead of just svector-var
 * pieces. In case of pure unstruct-variables (ucvar), which are at the
 * end of each part, we might need to modify initialize shell vector
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructVectorInitialize( HYPRE_SStructVector vector )
{
   HYPRE_Int               datasize;
   HYPRE_Int               nvars ;
   HYPRE_Int               nparts = hypre_SStructVectorNParts(vector) ;
   HYPRE_Int               var, part  ;
   HYPRE_Complex          *data ;
   HYPRE_Complex          *pdata ;
   HYPRE_Complex          *sdata  ;
   hypre_SStructPVector   *pvector;
   hypre_StructVector     *svector;
   HYPRE_Int              *dataindices;
   HYPRE_Int              *pdataindices;
   HYPRE_Int               vector_type = hypre_SStructVectorObjectType(vector);
   hypre_SStructGrid      *grid =  hypre_SStructVectorGrid(vector);
   MPI_Comm                comm = hypre_SStructVectorComm(vector);
   HYPRE_IJVector          ijvector;
   hypre_SStructPGrid     *pgrid;
   HYPRE_SStructVariable  *vartypes;
   HYPRE_MemoryLocation    memory_location = hypre_HandleMemoryLocation(hypre_handle());

   /* GEC0902 addition of variables for ilower and iupper   */
   HYPRE_BigInt            ilower, iupper;
   hypre_ParVector        *par_vector;
   hypre_Vector           *parlocal_vector;


   /* GEC0902 getting the datasizes and indices we need  */

   hypre_SStructVectorInitializeShell(vector);

   datasize = hypre_SStructVectorDataSize(vector);

   data = hypre_CTAlloc(HYPRE_Complex, datasize, memory_location);

   dataindices = hypre_SStructVectorDataIndices(vector);

   hypre_SStructVectorData(vector) = data;

   for (part = 0; part < nparts; part++)
   {
      pvector = hypre_SStructVectorPVector(vector, part);
      pdataindices = hypre_SStructPVectorDataIndices(pvector);
      /* shift-num   = dataindices[part]; */
      pdata = data + dataindices[part];
      nvars = hypre_SStructPVectorNVars(pvector);

      pgrid    = hypre_SStructPVectorPGrid(pvector);
      vartypes = hypre_SStructPGridVarTypes(pgrid);
      for (var = 0; var < nvars; var++)
      {
         svector = hypre_SStructPVectorSVector(pvector, var);
         /*  shift-pnum    = pdataindices[var]; */
         sdata   = pdata + pdataindices[var];

         /* GEC1002 initialization of inside data pointer of a svector
          * because no data is alloced, we make sure the flag is zero. This
          * affects the destroy */
         hypre_StructVectorInitializeData(svector, sdata);
         hypre_StructVectorDataAlloced(svector) = 0;
         if (vartypes[var] > 0)
         {
            /* needed to get AddTo accumulation correct between processors */
            hypre_StructVectorClearGhostValues(svector);
         }
      }
   }

   /* GEC1002 this is now the creation of the ijmatrix and the initialization
    * by checking the type of the vector */

   if (vector_type == HYPRE_PARCSR )
   {
      ilower = hypre_SStructGridStartRank(grid);
      iupper = ilower + hypre_SStructGridLocalSize(grid) - 1;
   }
   else if (vector_type == HYPRE_SSTRUCT || vector_type == HYPRE_STRUCT)
   {
      ilower = hypre_SStructGridGhstartRank(grid);
      iupper = ilower + hypre_SStructGridGhlocalSize(grid) - 1;
   }
   else
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Invalid vector type!\n");
      return hypre_error_flag;
   }

   HYPRE_IJVectorCreate(comm, ilower, iupper,
                        &hypre_SStructVectorIJVector(vector));

   /* GEC1002, once the partitioning is done, it is time for the actual
    * initialization                                                 */


   /* u-vector: the type is for the parvector inside the ijvector */

   ijvector = hypre_SStructVectorIJVector(vector);

   HYPRE_IJVectorSetObjectType(ijvector, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(ijvector);
   HYPRE_IJVectorGetObject(ijvector, (void **) &hypre_SStructVectorParVector(vector));

   /* GEC1002 for HYPRE_SSTRUCT type of vector, we do not need data allocated
    * inside the parvector piece of the structure. We make that pointer within
    * the localvector to point to the outside "data". Before redirecting the
    * local pointer to point to the true data chunk for HYPRE_SSTRUCT: we
    * destroy and assign.  We now have two entries of the data structure
    * pointing to the same chunk if we have a HYPRE_SSTRUCT vector We do not
    * need the IJVectorInitializePar, we have to undoit for the SStruct case in
    * a sense it is a desinitializepar */

   if (vector_type == HYPRE_SSTRUCT || vector_type == HYPRE_STRUCT)
   {
      par_vector = (hypre_ParVector *) hypre_IJVectorObject(ijvector);
      parlocal_vector = hypre_ParVectorLocalVector(par_vector);
      hypre_TFree(hypre_VectorData(parlocal_vector), hypre_VectorMemoryLocation(parlocal_vector));
      hypre_VectorData(parlocal_vector) = data ;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructVectorSetValues( HYPRE_SStructVector  vector,
                              HYPRE_Int            part,
                              HYPRE_Int           *index,
                              HYPRE_Int            var,
                              HYPRE_Complex       *value )
{
   HYPRE_Int             ndim    = hypre_SStructVectorNDim(vector);
   hypre_SStructPVector *pvector = hypre_SStructVectorPVector(vector, part);
   hypre_Index           cindex;

   hypre_CopyToCleanIndex(index, ndim, cindex);

   if (var < hypre_SStructPVectorNVars(pvector))
   {
      hypre_SStructPVectorSetValues(pvector, cindex, var, value, 0);
   }
   else
   {
      /* TODO */
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructVectorAddToValues( HYPRE_SStructVector  vector,
                                HYPRE_Int            part,
                                HYPRE_Int           *index,
                                HYPRE_Int            var,
                                HYPRE_Complex       *value )
{
   HYPRE_Int             ndim    = hypre_SStructVectorNDim(vector);
   hypre_SStructPVector *pvector = hypre_SStructVectorPVector(vector, part);
   hypre_Index           cindex;

   hypre_CopyToCleanIndex(index, ndim, cindex);

   if (var < hypre_SStructPVectorNVars(pvector))
   {
      hypre_SStructPVectorSetValues(pvector, cindex, var, value, 1);
   }
   else
   {
      /* TODO */
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/* ONLY3D - RDF: Why? */

HYPRE_Int
HYPRE_SStructVectorAddFEMValues( HYPRE_SStructVector  vector,
                                 HYPRE_Int            part,
                                 HYPRE_Int           *index,
                                 HYPRE_Complex       *values )
{
   HYPRE_Int           ndim         = hypre_SStructVectorNDim(vector);
   hypre_SStructGrid  *grid         = hypre_SStructVectorGrid(vector);
   HYPRE_Int           fem_nvars    = hypre_SStructGridFEMPNVars(grid, part);
   HYPRE_Int          *fem_vars     = hypre_SStructGridFEMPVars(grid, part);
   hypre_Index        *fem_offsets  = hypre_SStructGridFEMPOffsets(grid, part);
   HYPRE_Int           i, d, vindex[HYPRE_MAXDIM];

   /* Set one variable at a time */
   for (i = 0; i < fem_nvars; i++)
   {
      for (d = 0; d < ndim; d++)
      {
         /* note: these offsets are different from what the user passes in */
         vindex[d] = index[d] + hypre_IndexD(fem_offsets[i], d);
      }
      HYPRE_SStructVectorAddToValues(
         vector, part, vindex, fem_vars[i], &values[i]);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructVectorGetValues( HYPRE_SStructVector  vector,
                              HYPRE_Int            part,
                              HYPRE_Int           *index,
                              HYPRE_Int            var,
                              HYPRE_Complex       *value )
{
   HYPRE_Int             ndim    = hypre_SStructVectorNDim(vector);
   hypre_SStructPVector *pvector = hypre_SStructVectorPVector(vector, part);
   hypre_Index           cindex;

   hypre_CopyToCleanIndex(index, ndim, cindex);

   if (var < hypre_SStructPVectorNVars(pvector))
   {
      hypre_SStructPVectorGetValues(pvector, cindex, var, value);
   }
   else
   {
      /* TODO */
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/* ONLY3D - RDF: Why? */

HYPRE_Int
HYPRE_SStructVectorGetFEMValues( HYPRE_SStructVector  vector,
                                 HYPRE_Int            part,
                                 HYPRE_Int           *index,
                                 HYPRE_Complex       *values )
{
   HYPRE_Int             ndim         = hypre_SStructVectorNDim(vector);
   hypre_SStructGrid    *grid         = hypre_SStructVectorGrid(vector);
   hypre_SStructPVector *pvector      = hypre_SStructVectorPVector(vector, part);
   HYPRE_Int             fem_nvars    = hypre_SStructGridFEMPNVars(grid, part);
   HYPRE_Int            *fem_vars     = hypre_SStructGridFEMPVars(grid, part);
   hypre_Index          *fem_offsets  = hypre_SStructGridFEMPOffsets(grid, part);
   HYPRE_Int             i, d, vindex[HYPRE_MAXDIM];

   hypre_SetIndex(vindex, 0);
   for (i = 0; i < fem_nvars; i++)
   {
      for (d = 0; d < ndim; d++)
      {
         /* note: these offsets are different from what the user passes in */
         vindex[d] = index[d] + hypre_IndexD(fem_offsets[i], d);
      }
      hypre_SStructPVectorGetValues(pvector, vindex, fem_vars[i], &values[i]);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructVectorSetBoxValues( HYPRE_SStructVector  vector,
                                 HYPRE_Int            part,
                                 HYPRE_Int           *ilower,
                                 HYPRE_Int           *iupper,
                                 HYPRE_Int            var,
                                 HYPRE_Complex       *values )
{
   HYPRE_SStructVectorSetBoxValues2(vector, part, ilower, iupper, var,
                                    ilower, iupper, values);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructVectorAddToBoxValues( HYPRE_SStructVector  vector,
                                   HYPRE_Int            part,
                                   HYPRE_Int           *ilower,
                                   HYPRE_Int           *iupper,
                                   HYPRE_Int            var,
                                   HYPRE_Complex       *values )
{
   HYPRE_SStructVectorAddToBoxValues2(vector, part, ilower, iupper, var,
                                      ilower, iupper, values);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructVectorGetBoxValues(HYPRE_SStructVector  vector,
                                HYPRE_Int            part,
                                HYPRE_Int           *ilower,
                                HYPRE_Int           *iupper,
                                HYPRE_Int            var,
                                HYPRE_Complex       *values )
{
   HYPRE_SStructVectorGetBoxValues2(vector, part, ilower, iupper, var,
                                    ilower, iupper, values);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructVectorSetBoxValues2( HYPRE_SStructVector  vector,
                                  HYPRE_Int            part,
                                  HYPRE_Int           *ilower,
                                  HYPRE_Int           *iupper,
                                  HYPRE_Int            var,
                                  HYPRE_Int           *vilower,
                                  HYPRE_Int           *viupper,
                                  HYPRE_Complex       *values )
{
   hypre_SStructPVector *pvector = hypre_SStructVectorPVector(vector, part);
   hypre_Box            *set_box, *value_box;
   HYPRE_Int             d, ndim = hypre_SStructVectorNDim(vector);

   /* This creates boxes with zeroed-out extents */
   set_box = hypre_BoxCreate(ndim);
   value_box = hypre_BoxCreate(ndim);

   for (d = 0; d < ndim; d++)
   {
      hypre_BoxIMinD(set_box, d) = ilower[d];
      hypre_BoxIMaxD(set_box, d) = iupper[d];
      hypre_BoxIMinD(value_box, d) = vilower[d];
      hypre_BoxIMaxD(value_box, d) = viupper[d];
   }

   hypre_SStructPVectorSetBoxValues(pvector, set_box, var, value_box, values, 0);

   hypre_BoxDestroy(set_box);
   hypre_BoxDestroy(value_box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructVectorAddToBoxValues2( HYPRE_SStructVector  vector,
                                    HYPRE_Int            part,
                                    HYPRE_Int           *ilower,
                                    HYPRE_Int           *iupper,
                                    HYPRE_Int            var,
                                    HYPRE_Int           *vilower,
                                    HYPRE_Int           *viupper,
                                    HYPRE_Complex       *values )
{
   hypre_SStructPVector *pvector = hypre_SStructVectorPVector(vector, part);
   hypre_Box            *set_box, *value_box;
   HYPRE_Int             d, ndim = hypre_SStructVectorNDim(vector);

   /* This creates boxes with zeroed-out extents */
   set_box = hypre_BoxCreate(ndim);
   value_box = hypre_BoxCreate(ndim);

   for (d = 0; d < ndim; d++)
   {
      hypre_BoxIMinD(set_box, d) = ilower[d];
      hypre_BoxIMaxD(set_box, d) = iupper[d];
      hypre_BoxIMinD(value_box, d) = vilower[d];
      hypre_BoxIMaxD(value_box, d) = viupper[d];
   }

   hypre_SStructPVectorSetBoxValues(pvector, set_box, var, value_box, values, 1);

   hypre_BoxDestroy(set_box);
   hypre_BoxDestroy(value_box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructVectorGetBoxValues2(HYPRE_SStructVector  vector,
                                 HYPRE_Int            part,
                                 HYPRE_Int           *ilower,
                                 HYPRE_Int           *iupper,
                                 HYPRE_Int            var,
                                 HYPRE_Int           *vilower,
                                 HYPRE_Int           *viupper,
                                 HYPRE_Complex       *values )
{
   hypre_SStructPVector *pvector = hypre_SStructVectorPVector(vector, part);
   hypre_Box            *set_box, *value_box;
   HYPRE_Int             d, ndim = hypre_SStructVectorNDim(vector);

   /* This creates boxes with zeroed-out extents */
   set_box = hypre_BoxCreate(ndim);
   value_box = hypre_BoxCreate(ndim);

   for (d = 0; d < ndim; d++)
   {
      hypre_BoxIMinD(set_box, d) = ilower[d];
      hypre_BoxIMaxD(set_box, d) = iupper[d];
      hypre_BoxIMinD(value_box, d) = vilower[d];
      hypre_BoxIMaxD(value_box, d) = viupper[d];
   }

   hypre_SStructPVectorGetBoxValues(pvector, set_box, var, value_box, values);

   hypre_BoxDestroy(set_box);
   hypre_BoxDestroy(value_box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructVectorAddFEMBoxValues(HYPRE_SStructVector  vector,
                                   HYPRE_Int            part,
                                   HYPRE_Int           *ilower,
                                   HYPRE_Int           *iupper,
                                   HYPRE_Complex       *values)
{
   HYPRE_Int             ndim            = hypre_SStructVectorNDim(vector);
   hypre_SStructGrid    *grid            = hypre_SStructVectorGrid(vector);
   HYPRE_MemoryLocation  memory_location = hypre_SStructVectorMemoryLocation(vector);

   HYPRE_Int             fem_nvars       = hypre_SStructGridFEMPNVars(grid, part);
   HYPRE_Int            *fem_vars        = hypre_SStructGridFEMPVars(grid, part);
   hypre_Index          *fem_offsets     = hypre_SStructGridFEMPOffsets(grid, part);

   HYPRE_Complex        *tvalues;
   hypre_Box            *box;

   HYPRE_Int             i, d, vilower[HYPRE_MAXDIM], viupper[HYPRE_MAXDIM];
   HYPRE_Int             ei, vi, nelts;

   /* Set one variable at a time */
   box = hypre_BoxCreate(ndim);
   hypre_BoxSetExtents(box, ilower, iupper);
   nelts = hypre_BoxVolume(box);
   tvalues = hypre_TAlloc(HYPRE_Complex, nelts, memory_location);

   for (i = 0; i < fem_nvars; i++)
   {
      for (d = 0; d < ndim; d++)
      {
         /* note: these offsets are different from what the user passes in */
         vilower[d] = ilower[d] + hypre_IndexD(fem_offsets[i], d);
         viupper[d] = iupper[d] + hypre_IndexD(fem_offsets[i], d);
      }

#if defined(HYPRE_USING_GPU)
      if (hypre_GetExecPolicy1(memory_location) == HYPRE_EXEC_DEVICE)
      {
         hypreDevice_ComplexStridedCopy(nelts, fem_nvars, values + i, tvalues);
      }
      else
#endif
      {
         for (ei = 0, vi = i; ei < nelts; ei ++, vi += fem_nvars)
         {
            tvalues[ei] = values[vi];
         }
      }

      HYPRE_SStructVectorAddToBoxValues(vector, part, vilower, viupper,
                                        fem_vars[i], tvalues);
   }

   /* Free memory */
   hypre_TFree(tvalues, memory_location);
   hypre_BoxDestroy(box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructVectorAssemble( HYPRE_SStructVector vector )
{
   hypre_SStructGrid      *grid            = hypre_SStructVectorGrid(vector);
   HYPRE_Int               nparts          = hypre_SStructVectorNParts(vector);
   HYPRE_IJVector          ijvector        = hypre_SStructVectorIJVector(vector);
   hypre_SStructCommInfo **vnbor_comm_info = hypre_SStructGridVNborCommInfo(grid);
   HYPRE_Int               vnbor_ncomms    = hypre_SStructGridVNborNComms(grid);
   HYPRE_Int               part;

   hypre_CommInfo         *comm_info;
   HYPRE_Int               send_part,    recv_part;
   HYPRE_Int               send_var,     recv_var;
   hypre_StructVector     *send_vector, *recv_vector;
   hypre_CommPkg          *comm_pkg;
   hypre_CommHandle       *comm_handle;
   HYPRE_Int               ci;

   /*------------------------------------------------------
    * Communicate and accumulate within parts
    *------------------------------------------------------*/

   for (part = 0; part < nparts; part++)
   {
      hypre_SStructPVectorAccumulate(hypre_SStructVectorPVector(vector, part));
   }

   /*------------------------------------------------------
    * Communicate and accumulate between parts
    *------------------------------------------------------*/

   for (ci = 0; ci < vnbor_ncomms; ci++)
   {
      comm_info = hypre_SStructCommInfoCommInfo(vnbor_comm_info[ci]);
      send_part = hypre_SStructCommInfoSendPart(vnbor_comm_info[ci]);
      recv_part = hypre_SStructCommInfoRecvPart(vnbor_comm_info[ci]);
      send_var  = hypre_SStructCommInfoSendVar(vnbor_comm_info[ci]);
      recv_var  = hypre_SStructCommInfoRecvVar(vnbor_comm_info[ci]);

      send_vector = hypre_SStructPVectorSVector(
                       hypre_SStructVectorPVector(vector, send_part), send_var);
      recv_vector = hypre_SStructPVectorSVector(
                       hypre_SStructVectorPVector(vector, recv_part), recv_var);

      /* want to communicate and add ghost data to real data */
      hypre_CommPkgCreate(comm_info,
                          hypre_StructVectorDataSpace(send_vector),
                          hypre_StructVectorDataSpace(recv_vector),
                          1, NULL, 1, hypre_StructVectorComm(send_vector),
                          &comm_pkg);
      /* note reversal of send/recv data here */
      hypre_InitializeCommunication(comm_pkg,
                                    hypre_StructVectorData(recv_vector),
                                    hypre_StructVectorData(send_vector),
                                    1, 0, &comm_handle);
      hypre_FinalizeCommunication(comm_handle);
      hypre_CommPkgDestroy(comm_pkg);
   }

   /*------------------------------------------------------
    * Assemble P and U vectors
    *------------------------------------------------------*/

   for (part = 0; part < nparts; part++)
   {
      hypre_SStructPVectorAssemble(hypre_SStructVectorPVector(vector, part));
   }

   /* u-vector */
   HYPRE_IJVectorAssemble(ijvector);

   /*------------------------------------------------------
    *------------------------------------------------------*/

   /* if the object type is parcsr, then convert the sstruct vector which has ghost
      layers to a parcsr vector without ghostlayers. */
   if (hypre_SStructVectorObjectType(vector) == HYPRE_PARCSR)
   {
      hypre_SStructVectorParConvert(vector,
                                    &hypre_SStructVectorParVector(vector));
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * RDF: I don't think this will work correctly in the case where a processor's
 * data is shared entirely with other processors.  The code in PGridAssemble
 * ensures that data is uniquely distributed, so the data box for this processor
 * would be empty and there would be no ghost zones to fill in Gather.
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructVectorGather( HYPRE_SStructVector vector )
{
   hypre_SStructGrid      *grid            = hypre_SStructVectorGrid(vector);
   HYPRE_Int               nparts          = hypre_SStructVectorNParts(vector);
   hypre_SStructCommInfo **vnbor_comm_info = hypre_SStructGridVNborCommInfo(grid);
   HYPRE_Int               vnbor_ncomms    = hypre_SStructGridVNborNComms(grid);
   HYPRE_Int               part;

   hypre_CommInfo         *comm_info;
   HYPRE_Int               send_part,    recv_part;
   HYPRE_Int               send_var,     recv_var;
   hypre_StructVector     *send_vector, *recv_vector;
   hypre_CommPkg          *comm_pkg;
   hypre_CommHandle       *comm_handle;
   HYPRE_Int               ci;

   /* GEC1102 we change the name of the restore-->parrestore  */

   if (hypre_SStructVectorObjectType(vector) == HYPRE_PARCSR)
   {
      hypre_SStructVectorParRestore(vector, hypre_SStructVectorParVector(vector));
   }

   for (part = 0; part < nparts; part++)
   {
      hypre_SStructPVectorGather(hypre_SStructVectorPVector(vector, part));
   }

   /* gather shared data from other parts */

   for (ci = 0; ci < vnbor_ncomms; ci++)
   {
      comm_info = hypre_SStructCommInfoCommInfo(vnbor_comm_info[ci]);
      send_part = hypre_SStructCommInfoSendPart(vnbor_comm_info[ci]);
      recv_part = hypre_SStructCommInfoRecvPart(vnbor_comm_info[ci]);
      send_var  = hypre_SStructCommInfoSendVar(vnbor_comm_info[ci]);
      recv_var  = hypre_SStructCommInfoRecvVar(vnbor_comm_info[ci]);

      send_vector = hypre_SStructPVectorSVector(
                       hypre_SStructVectorPVector(vector, send_part), send_var);
      recv_vector = hypre_SStructPVectorSVector(
                       hypre_SStructVectorPVector(vector, recv_part), recv_var);

      /* want to communicate real data to ghost data */
      hypre_CommPkgCreate(comm_info,
                          hypre_StructVectorDataSpace(send_vector),
                          hypre_StructVectorDataSpace(recv_vector),
                          1, NULL, 0, hypre_StructVectorComm(send_vector),
                          &comm_pkg);
      hypre_InitializeCommunication(comm_pkg,
                                    hypre_StructVectorData(send_vector),
                                    hypre_StructVectorData(recv_vector),
                                    0, 0, &comm_handle);
      hypre_FinalizeCommunication(comm_handle);
      hypre_CommPkgDestroy(comm_pkg);

      /* boundary ghost values may not be clear */
      hypre_StructVectorBGhostNotClear(recv_vector) = 1;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructVectorSetConstantValues( HYPRE_SStructVector vector,
                                      HYPRE_Complex       value )
{
   hypre_SStructPVector *pvector;
   HYPRE_Int part;
   HYPRE_Int nparts   = hypre_SStructVectorNParts(vector);

   for ( part = 0; part < nparts; part++ )
   {
      pvector = hypre_SStructVectorPVector( vector, part );
      hypre_SStructPVectorSetConstantValues( pvector, value );
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructVectorSetObjectType( HYPRE_SStructVector  vector,
                                  HYPRE_Int            type )
{
   /* this implements only HYPRE_PARCSR, which is always available */
   hypre_SStructVectorObjectType(vector) = type;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructVectorGetObject( HYPRE_SStructVector   vector,
                              void                **object )
{
   HYPRE_Int             type = hypre_SStructVectorObjectType(vector);
   hypre_SStructPVector *pvector;
   hypre_StructVector   *svector;
   HYPRE_Int             part, var;

   if (type == HYPRE_SSTRUCT)
   {
      *object = vector;
   }
   else if (type == HYPRE_PARCSR)
   {
      *object = hypre_SStructVectorParVector(vector);
   }
   else if (type == HYPRE_STRUCT)
   {
      /* only one part & one variable */
      part = 0;
      var = 0;
      pvector = hypre_SStructVectorPVector(vector, part);
      svector = hypre_SStructPVectorSVector(pvector, var);
      *object = svector;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructVectorPrint
 *
 * This function prints a SStructVector to file. For the assumptions used
 * here, see HYPRE_SStructMatrixPrint.
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructVectorPrint( const char          *filename,
                          HYPRE_SStructVector  vector,
                          HYPRE_Int            all )
{
   /* Vector variables */
   MPI_Comm              comm = hypre_SStructVectorComm(vector);
   HYPRE_Int             nparts = hypre_SStructVectorNParts(vector);
   hypre_SStructGrid    *grid = hypre_SStructVectorGrid(vector);

   /* Local variables */
   hypre_SStructPVector *pvector;
   hypre_StructVector   *svector;

   FILE                 *file;
   HYPRE_Int             myid;
   HYPRE_Int             part, var, nvars;
   char                  new_filename[255];

   /* Print auxiliary data */
   hypre_MPI_Comm_rank(comm, &myid);
   hypre_sprintf(new_filename, "%s.%05d", filename, myid);
   if ((file = fopen(new_filename, "w")) == NULL)
   {
      hypre_printf("Error: can't open output file %s\n", new_filename);
      hypre_error_in_arg(1);

      return hypre_error_flag;
   }

   hypre_fprintf(file, "SStructVector\n");
   hypre_SStructGridPrint(file, grid);

   /* Print (part, var) vectors */
   for (part = 0; part < nparts; part++)
   {
      pvector = hypre_SStructVectorPVector(vector, part);
      nvars = hypre_SStructPVectorNVars(pvector);
      for (var = 0; var < nvars; var++)
      {
         svector = hypre_SStructPVectorSVector(pvector, var);

         hypre_fprintf(file, "\nData - (Part %d, Var %d):\n", part, var);
         hypre_StructVectorPrintData(file, svector, all);
      }
   }

   fclose(file);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructVectorRead
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructVectorRead( MPI_Comm             comm,
                         const char          *filename,
                         HYPRE_SStructVector *vector_ptr )
{
   /* Vector variables */
   HYPRE_SStructVector    vector;
   hypre_SStructPVector  *pvector;
   hypre_StructVector    *svector;
   hypre_SStructGrid     *grid;
   HYPRE_Int              nparts;
   HYPRE_Int              nvars;

   /* Local variables */
   FILE                  *file;
   char                   new_filename[255];
   HYPRE_Int              p, v, part, var;
   HYPRE_Int              myid;

   /* Read auxiliary data */
   hypre_MPI_Comm_rank(comm, &myid);
   hypre_sprintf(new_filename, "%s.%05d", filename, myid);
   if ((file = fopen(new_filename, "r")) == NULL)
   {
      hypre_printf("Error: can't open input file %s\n", new_filename);
      hypre_error_in_arg(2);

      return hypre_error_flag;
   }

   hypre_fscanf(file, "SStructVector\n");
   hypre_SStructGridRead(comm, file, &grid);

   /* Create and initialize vector */
   HYPRE_SStructVectorCreate(comm, grid, &vector);
   HYPRE_SStructVectorInitialize(vector);

   /* Read values from file */
   nparts = hypre_SStructVectorNParts(vector);
   for (p = 0; p < nparts; p++)
   {
      pvector = hypre_SStructVectorPVector(vector, p);
      nvars = hypre_SStructPVectorNVars(pvector);

      for (v = 0; v < nvars; v++)
      {
         hypre_fscanf(file, "\nData - (Part %d, Var %d):\n", &part, &var);

         pvector = hypre_SStructVectorPVector(vector, part);
         svector = hypre_SStructPVectorSVector(pvector, var);

         hypre_StructVectorReadData(file, svector);
      }
   }
   fclose(file);

   /* Assemble vector */
   HYPRE_SStructVectorAssemble(vector);

   /* Decrease ref counters */
   HYPRE_SStructGridDestroy(grid);

   *vector_ptr = vector;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * copy x to y, y should already exist and be the same size
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructVectorCopy( HYPRE_SStructVector x,
                         HYPRE_SStructVector y )
{
   hypre_SStructCopy(x, y);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * y = a*y, for vector y and scalar a
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructVectorScale( HYPRE_Complex       alpha,
                          HYPRE_SStructVector y )
{
   hypre_SStructScale( alpha, (hypre_SStructVector *)y );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * inner or dot product, result = < x, y >
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructInnerProd( HYPRE_SStructVector x,
                        HYPRE_SStructVector y,
                        HYPRE_Real         *result )
{
   hypre_SStructInnerProd(x, y, result);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * y = y + alpha*x for vectors y, x and scalar alpha
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructAxpy( HYPRE_Complex       alpha,
                   HYPRE_SStructVector x,
                   HYPRE_SStructVector y )
{
   hypre_SStructAxpy(alpha, x, y);

   return hypre_error_flag;
}
