/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

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
#define hypre_SchwarzDataVtemp(schwarz_data) ((schwarz_data)->Vtemp)
#define hypre_SchwarzDataScale(schwarz_data) ((schwarz_data)->scale)
#define hypre_SchwarzDataDofFunc(schwarz_data) ((schwarz_data)->dof_func)

#endif



