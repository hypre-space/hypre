/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/

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

   int                     nvars;
   hypre_StructVector    **svectors;     /* nvar array of svectors */
   hypre_CommPkg         **comm_pkgs;    /* nvar array of comm pkgs */

   int                     complex;      /* Is the vector complex */

   int                     ref_count;

   int                    *dataindices;  /* GEC1002 array for starting index of the 
                                            svector. pdataindices[varx] */
   int                     datasize;     /* Size of the pvector = sums size of svectors */

} hypre_SStructPVector;

typedef struct hypre_SStructVector_struct
{
   MPI_Comm                comm;
   int                     ndim;
   hypre_SStructGrid      *grid;
   int                     object_type;

   /* s-vector info */
   int                     nparts;
   hypre_SStructPVector  **pvectors;
   hypre_CommPkg        ***comm_pkgs;    /* nvar array of comm pkgs */

   /* u-vector info */
   HYPRE_IJVector          ijvector;
   hypre_ParVector        *parvector;

  /* GEC10020902 pointer to big chunk of memory and auxiliary information   */

   double                  *data;        /* GEC1002 pointer to chunk data  */
   int                     *dataindices; /* GEC1002 dataindices[partx] is the starting index
                                          of vector data for the part=partx    */
   int                     datasize    ;  /* GEC1002 size of all data = ghlocalsize */

   int                     complex;      /* Is the vector complex */
   int                     global_size;  /* Total number coefficients */

   int                     ref_count;

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
#define hypre_SStructVectorComplex(vec)        ((vec) -> complex)
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
#define hypre_SStructPVectorComplex(pvec)     ((pvec) -> complex)
#define hypre_SStructPVectorRefCount(pvec)    ((pvec) -> ref_count)
#define hypre_SStructPVectorDataIndices(pvec) ((pvec) -> dataindices  )
#define hypre_SStructPVectorDataSize(pvec)    ((pvec) -> datasize  )

#endif
