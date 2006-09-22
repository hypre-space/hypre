/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/




#ifndef hypre_Schwarz_DATA_HEADER
#define hypre_Schwarz_DATA_HEADER

/*--------------------------------------------------------------------------
 * hypre_SchwarzData
 *--------------------------------------------------------------------------*/

typedef struct
{
   int      variant;
   int      domain_type;
   int      overlap;
   int      num_functions;
   double   relax_weight;

   hypre_CSRMatrix *domain_structure;
   hypre_CSRMatrix *A_boundary;
   hypre_ParVector *Vtemp;
   double  *scale;
   int     *dof_func;

} hypre_SchwarzData;

/*--------------------------------------------------------------------------
 * Accessor functions for the hypre_SchwarzData structure
 *--------------------------------------------------------------------------*/

#define hypre_SchwarzDataVariant(schwarz_data) ((schwarz_data)->variant)
#define hypre_SchwarzDataDomainType(schwarz_data) ((schwarz_data)->domain_type)
#define hypre_SchwarzDataOverlap(schwarz_data) ((schwarz_data)->overlap)
#define hypre_SchwarzDataNumFunctions(schwarz_data) \
((schwarz_data)->num_functions)
#define hypre_SchwarzDataRelaxWeight(schwarz_data) \
((schwarz_data)->relax_weight)
#define hypre_SchwarzDataDomainStructure(schwarz_data) \
((schwarz_data)->domain_structure)
#define hypre_SchwarzDataABoundary(schwarz_data) ((schwarz_data)->A_boundary)
#define hypre_SchwarzDataVtemp(schwarz_data) ((schwarz_data)->Vtemp)
#define hypre_SchwarzDataScale(schwarz_data) ((schwarz_data)->scale)
#define hypre_SchwarzDataDofFunc(schwarz_data) ((schwarz_data)->dof_func)

#endif



