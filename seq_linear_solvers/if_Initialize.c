#include "general.h"

#include "incfact_data.h"

/*--------------------------------------------------------------------------
 * Purpose:       Allocate and initialize the data structure containing all
 *                information for incomplete factorization and solve. Sets
 *                default values for parameters used in numerical routines.
 *                These parameters can be accessed via the routines in
 *                GetINCFACTDataIpar.c.
 *                Also included is a routine for freeing data structure.

 * Author:        Andy Cleary
 *                Centre for Applied Scientific Computing
 *                Lawrence Livermore Labs
 * Revision History:
 *  5-22-97: AJC. Original Version.
 *--------------------------------------------------------------------------*/


/*--------------------------------------------------------------------------
 * incfact_Initialize
 *--------------------------------------------------------------------------*/

void  *incfact_initialize( port )
void  *port;
{
   INCFACTData    *incfact_data;
   int             i;


   /* allocate space for structure */
   incfact_data = ctalloc(INCFACTData, 1);

   /* Control variable array IPAR */

   INCFACTDataIpar(incfact_data)[0] = 1; /* Reordering?: default is no */
   INCFACTDataIpar(incfact_data)[1] = 3; /* Scaling?: default is no */
   INCFACTDataIpar(incfact_data)[2] = 0; /* Output message device; default is iout=0 */
   INCFACTDataIpar(incfact_data)[3] = 0; /* Not used. */
   INCFACTDataIpar(incfact_data)[4] = 0; /* Not used. */
   INCFACTDataIpar(incfact_data)[5] = 20;/* lfil_incfactt: # of els per row in factor. 
                                            No Default */
#ifdef ILUFact
   INCFACTDataIpar(incfact_data)[6] = 25;/* Dimension of KSP subspace. No Default */
#endif
#ifdef ICFact
   INCFACTDataIpar(incfact_data)[6] = 0;/* Not used. */
#endif
   INCFACTDataIpar(incfact_data)[7] = 0; /* Maxits for KSP. Default (100)*/
   
   for( i=8; i<20; i++ )
     INCFACTDataIpar(incfact_data)[i] = 0; /* unused */
 
   /* Control variable array RPAR */

   INCFACTDataRpar(incfact_data)[0] = 0.0; /* tol_incfactt, drop tol. Default (0.0001) */
   INCFACTDataRpar(incfact_data)[1] = 0.0000000001; /* KSP stopping criterion; default(.00001) */

   /* Default mode is that input matrix cannot be overwritten */
   INCFACTDataMode(incfact_data) = 1;

   return( (void *) incfact_data );
}

void     incfact_free(incfact_data)
INCFACTData  *incfact_data;
{

   if (incfact_data)
     {

     tfree(INCFACTDataPerm(incfact_data));
     tfree(INCFACTDataInversePerm(incfact_data));
#ifdef ILUFact
     tfree(INCFACTDataRscale(incfact_data));
     tfree(INCFACTDataCscale(incfact_data));
#endif
#ifdef ICFact
     tfree(INCFACTDataScale(incfact_data));
#endif
     tfree(INCFACTDataIwork(incfact_data));
     tfree(INCFACTDataRwork(incfact_data));

     FreeMatrix(INCFACTDataPreconditioner(incfact_data));

     tfree(incfact_data);
     }
}
