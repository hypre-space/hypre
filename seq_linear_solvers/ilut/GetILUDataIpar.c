#include "general.h"

#include "ilu_data.h"

/*--------------------------------------------------------------------------
 * Purpose:       Routines for setting parameters to incomplete 
 *                factorization. Applications should use these routines
 *                to set the appropriate parameters so that they do not
 *                have to have any knowledge or dependence on the data
 *                structures internal to the factor and solve routines.

 * Author:        Andy Cleary
 *                Centre for Applied Scientific Computing
 *                Lawrence Livermore Labs
 * Revision History:
 *  5-22-97: AJC. Original Version.
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * GetILUDataIpar
 *--------------------------------------------------------------------------*/

int *GetILUDataIpar( void * data )
     /* Returns a pointer to the beginning of the integer vector that holds
        integer parameters (many optional) to the incomplete factorization
        and solve codes.
        For a description of each parameter and its default values, please
        see ilu_initialize.
        This routine should be called *after* ilut_initialize, but
        *before* ilu_setup.
        Element i of the IPAR vector can be accessed as
        ILUDataIpar(data)[i],
        where "data" is the void * pointer to the solver structure.
      */
{
   ILUData    *ilu_data;

   ilu_data = (ILUData *) data;

   return( ILUDataIpar(ilu_data) );
}

/*--------------------------------------------------------------------------
 * GetILUDataRpar
 *--------------------------------------------------------------------------*/

double *GetILUDataRpar( void * data )
     /* Returns a pointer to the beginning of the vector of doubles that holds
        floating point parameters (some optional) to the incomplete factorization
        and solve codes.
        For a description of each parameter and its default values, please
        see ilut_initialize.
        This routine should be called *after* ilut_initialize, but
        *before* ilu_setup.
        Element i of the RPAR vector can be accessed as
        ILUDataRpar(data)[i],
        where "data" is the void * pointer to the solver structure.
      */

{
   ILUData    *ilu_data;

   ilu_data = (ILUData *) data;

   return( ILUDataRpar(ilu_data) );
}

