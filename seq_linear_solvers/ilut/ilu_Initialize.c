#include "general.h"

#include "ilu_data.h"

/*--------------------------------------------------------------------------
 * Purpose:       Allocate and initialize the data structure containing all
 *                information for incomplete factorization and solve. Sets
 *                default values for parameters used in numerical routines.
 *                These parameters can be accessed via the routines in
 *                GetILUDataIpar.c.
 *                Also included is a routine for freeing data structure.

 * Author:        Andy Cleary
 *                Centre for Applied Scientific Computing
 *                Lawrence Livermore Labs
 * Revision History:
 *  5-22-97: AJC. Original Version.
 *--------------------------------------------------------------------------*/


/*--------------------------------------------------------------------------
 * ilu_Initialize
 *--------------------------------------------------------------------------*/

void  *ilu_initialize( port )
void  *port;
{
   ILUData    *ilu_data;
   int             i;


   /* allocate space for structure */
   ilu_data = ctalloc(ILUData, 1);

   /* Control variable array IPAR */

   ILUDataIpar(ilu_data)[0] = 1; /* Reordering?: default is no */
   ILUDataIpar(ilu_data)[1] = 3; /* Scaling?: default is no */
   ILUDataIpar(ilu_data)[2] = 0; /* Output message device; default is iout=0 */
   ILUDataIpar(ilu_data)[3] = 0; /* Not used. */
   ILUDataIpar(ilu_data)[4] = 0; /* Not used. */
   ILUDataIpar(ilu_data)[5] = 20;/* lfil_ilut: # of els per row in factor. 
                                            No Default */
#ifdef ILUFact
   ILUDataIpar(ilu_data)[6] = 25;/* Dimension of GMRES subspace. No Default */
#endif
#ifdef ICFact
   ILUDataIpar(ilu_data)[6] = 0;/* Not used. */
#endif
   ILUDataIpar(ilu_data)[7] = 0; /* Maxits for GMRES. Default (100)*/
   
   for( i=8; i<20; i++ )
     ILUDataIpar(ilu_data)[i] = 0; /* unused */
 
   /* Control variable array RPAR */

   ILUDataRpar(ilu_data)[0] = 0.0; /* tol_ilut, drop tol. Default (0.0001) */
   ILUDataRpar(ilu_data)[1] = 0.0000000001; /* GMRES stopping criterion; default(.00001) */

   /* Default mode is that input matrix cannot be overwritten */
   ILUDataMode(ilu_data) = 1;

   return( (void *) ilu_data );
}

void     ilu_free(ilu_data)
ILUData  *ilu_data;
{

   if (ilu_data)
     {

     tfree(ILUDataPerm(ilu_data));
     tfree(ILUDataInversePerm(ilu_data));
#ifdef ILUFact
     tfree(ILUDataRscale(ilu_data));
     tfree(ILUDataCscale(ilu_data));
#endif
#ifdef ICFact
     tfree(ILUDataScale(ilu_data));
#endif
     tfree(ILUDataIwork(ilu_data));
     tfree(ILUDataRwork(ilu_data));

     FreeMatrix(ILUDataPreconditioner(ilu_data));

     tfree(ilu_data);
     }
}
