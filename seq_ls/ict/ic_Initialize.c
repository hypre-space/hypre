#include "general.h"

#include "ic_data.h"

/*--------------------------------------------------------------------------
 * Purpose:       Allocate and initialize the data structure containing all
 *                information for incomplete factorization and solve. Sets
 *                default values for parameters used in numerical routines.
 *                These parameters can be accessed via the routines in
 *                GetICDataIpar.c.
 *                Also included is a routine for freeing data structure.

 * Author:        Andy Cleary
 *                Centre for Applied Scientific Computing
 *                Lawrence Livermore Labs
 * Revision History:
 *  5-22-97: AJC. Original Version.
 *--------------------------------------------------------------------------*/


/*--------------------------------------------------------------------------
 * ic_Initialize
 *--------------------------------------------------------------------------*/

void  *ic_initialize( port )
void  *port;
{
   ICData    *ic_data;
   int             i;


   /* allocate space for structure */
   ic_data = ctalloc(ICData, 1);

   /* Control variable array IPAR */

   ICDataIpar(ic_data)[0] = 1; /* Reordering?: default is no */
   ICDataIpar(ic_data)[1] = 3; /* Scaling?: default is no */
   ICDataIpar(ic_data)[2] = 0; /* Output message device; default is iout=0 */
   ICDataIpar(ic_data)[3] = 0; /* Not used. */
   ICDataIpar(ic_data)[4] = 0; /* Not used. */
   ICDataIpar(ic_data)[5] = 20;/* lfil_ict: # of els per row in factor. 
                                            No Default */
#ifdef ILUFact
   ICDataIpar(ic_data)[6] = 25;/* Dimension of CG subspace. No Default */
#endif
#ifdef ICFact
   ICDataIpar(ic_data)[6] = 0;/* Not used. */
#endif
   ICDataIpar(ic_data)[7] = 0; /* Maxits for CG. Default (100)*/
   
   for( i=8; i<20; i++ )
     ICDataIpar(ic_data)[i] = 0; /* unused */
 
   /* Control variable array RPAR */

   ICDataRpar(ic_data)[0] = 0.0; /* tol_ict, drop tol. Default (0.0001) */
   ICDataRpar(ic_data)[1] = 0.0000000001; /* CG stopping criterion; default(.00001) */

   /* Default mode is that input matrix cannot be overwritten */
   ICDataMode(ic_data) = 1;

   return( (void *) ic_data );
}

void     ic_free(ic_data)
ICData  *ic_data;
{

   if (ic_data)
     {

     tfree(ICDataPerm(ic_data));
     tfree(ICDataInversePerm(ic_data));
#ifdef ILUFact
     tfree(ICDataRscale(ic_data));
     tfree(ICDataCscale(ic_data));
#endif
#ifdef ICFact
     tfree(ICDataScale(ic_data));
#endif
     tfree(ICDataIwork(ic_data));
     tfree(ICDataRwork(ic_data));

     FreeMatrix(ICDataPreconditioner(ic_data));

     tfree(ic_data);
     }
}
