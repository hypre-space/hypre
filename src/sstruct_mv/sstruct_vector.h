/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for the hypre_SStructVector structures
 *
 *****************************************************************************/

#ifndef hypre_SSTRUCT_VECTOR_HEADER
#define hypre_SSTRUCT_VECTOR_HEADER

/*--------------------------------------------------------------------------
 * hypre_SStructVector:
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm                comm;
   hypre_SStructPGrid     *pgrid;

   HYPRE_Int               nvars;
   hypre_StructVector    **svectors;     /* nvar array of svectors */
   hypre_CommPkg         **comm_pkgs;    /* nvar array of comm pkgs */

   HYPRE_Int               accumulated;  /* AddTo values accumulated? */

   HYPRE_Int               ref_count;

   HYPRE_Int              *dataindices;  /* GEC1002 array for starting index of the
                                            svector. pdataindices[varx] */
   HYPRE_Int               datasize;     /* Size of the pvector = sums size of svectors */

} hypre_SStructPVector;

typedef struct hypre_SStructVector_struct
{
   MPI_Comm                comm;
   HYPRE_Int               ndim;
   hypre_SStructGrid      *grid;
   HYPRE_Int               object_type;

   /* s-vector info */
   HYPRE_Int               nparts;
   hypre_SStructPVector  **pvectors;
   hypre_CommPkg        ***comm_pkgs;    /* nvar array of comm pkgs */

   /* u-vector info */
   HYPRE_IJVector          ijvector;
   hypre_ParVector        *parvector;

   /* inter-part communication info */
   HYPRE_Int               nbor_ncomms;  /* num comm_pkgs with neighbor parts */

   /* GEC10020902 pointer to big chunk of memory and auxiliary information */
   HYPRE_Complex          *data;        /* GEC1002 pointer to chunk data */
   HYPRE_Int              *dataindices; /* GEC1002 dataindices[partx] is the starting index
                                           of vector data for the part=partx */
   HYPRE_Int               datasize;    /* GEC1002 size of all data = ghlocalsize */

   HYPRE_Int               global_size;  /* Total number coefficients */
   HYPRE_Int               ref_count;

} hypre_SStructVector;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SStructVector
 *--------------------------------------------------------------------------*/

#define hypre_SStructVectorComm(vec)           ((vec) -> comm)
#define hypre_SStructVectorNDim(vec)           ((vec) -> ndim)
#define hypre_SStructVectorGrid(vec)           ((vec) -> grid)
#define hypre_SStructVectorObjectType(vec)     ((vec) -> object_type)
#define hypre_SStructVectorNParts(vec)         ((vec) -> nparts)
#define hypre_SStructVectorPVectors(vec)       ((vec) -> pvectors)
#define hypre_SStructVectorPVector(vec, part)  ((vec) -> pvectors[part])
#define hypre_SStructVectorIJVector(vec)       ((vec) -> ijvector)
#define hypre_SStructVectorParVector(vec)      ((vec) -> parvector)
#define hypre_SStructVectorNborNComms(vec)     ((vec) -> nbor_ncomms)
#define hypre_SStructVectorGlobalSize(vec)     ((vec) -> global_size)
#define hypre_SStructVectorRefCount(vec)       ((vec) -> ref_count)
#define hypre_SStructVectorData(vec)           ((vec) -> data )
#define hypre_SStructVectorDataIndices(vec)    ((vec) -> dataindices)
#define hypre_SStructVectorDataSize(vec)       ((vec) -> datasize)


/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SStructPVector
 *--------------------------------------------------------------------------*/

#define hypre_SStructPVectorComm(pvec)        ((pvec) -> comm)
#define hypre_SStructPVectorPGrid(pvec)       ((pvec) -> pgrid)
#define hypre_SStructPVectorNVars(pvec)       ((pvec) -> nvars)
#define hypre_SStructPVectorSVectors(pvec)    ((pvec) -> svectors)
#define hypre_SStructPVectorSVector(pvec, v)  ((pvec) -> svectors[v])
#define hypre_SStructPVectorCommPkgs(pvec)    ((pvec) -> comm_pkgs)
#define hypre_SStructPVectorCommPkg(pvec, v)  ((pvec) -> comm_pkgs[v])
#define hypre_SStructPVectorAccumulated(pvec) ((pvec) -> accumulated)
#define hypre_SStructPVectorRefCount(pvec)    ((pvec) -> ref_count)
#define hypre_SStructPVectorDataIndices(pvec) ((pvec) -> dataindices  )
#define hypre_SStructPVectorDataSize(pvec)    ((pvec) -> datasize  )

#endif
