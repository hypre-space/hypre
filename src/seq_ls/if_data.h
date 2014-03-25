/*--------------------------------------------------------------------------
 * Purpose:       Define data structure and access macros for incomplete
 *                factorization data, including the matrix, the factors of
 *                the matrix, and parameters used in the factorization and
 *                solve.

 * Author:        Andy Cleary
 *                Centre for Applied Scientific Computing
 *                Lawrence Livermore Labs
 * Revision History:
 *  5-22-97: AJC. Original Version.
 *--------------------------------------------------------------------------*/

#ifndef _INCFACTDATA_HEADER
#define _INCFACTDATA_HEADER

#define INCFACTFact

#ifdef ICFact
  #define rscale scale
#endif

#include "matrix.h"
/*--------------------------------------------------------------------------
 * incfact_data
 *--------------------------------------------------------------------------*/

typedef struct
{

  HYPRE_Int             mode;
  Matrix         *A;
  HYPRE_Int             ipar[20];
  HYPRE_Real      rpar[20];
  Matrix         *preconditioner;
  HYPRE_Int             lenpmx;
  HYPRE_Int            *perm;
  HYPRE_Int            *inverse_perm;
  HYPRE_Real     *rscale;
#ifdef ILUFact
  HYPRE_Real     *cscale;
#endif
  HYPRE_Int            *iwork;
  HYPRE_Int             l_iwork;
  HYPRE_Real     *rwork;
  HYPRE_Int             l_rwork;
  
} INCFACTData;

/*--------------------------------------------------------------------------
 * Accessor functions for the incfact_data structure
 *--------------------------------------------------------------------------*/
#define INCFACTDataMode(incfact_data)               ((incfact_data) -> mode)
#define INCFACTDataA(incfact_data)                  ((incfact_data) -> A)
#define INCFACTDataIpar(incfact_data)               ((incfact_data) -> ipar)
#define INCFACTDataRpar(incfact_data)               ((incfact_data) -> rpar)
#define INCFACTDataPreconditioner(incfact_data)     ((incfact_data) -> preconditioner)
#define INCFACTDataLenpmx(incfact_data)             ((incfact_data) -> lenpmx)
#define INCFACTDataPerm(incfact_data)               ((incfact_data) -> perm)
#define INCFACTDataInversePerm(incfact_data)        ((incfact_data) -> inverse_perm)
#ifdef ILUFact
  #define INCFACTDataRscale(incfact_data)             ((incfact_data) -> rscale)
  #define INCFACTDataCscale(incfact_data)             ((incfact_data) -> cscale)
#else
  #define INCFACTDataScale(incfact_data)              ((incfact_data) -> scale)
  #define INCFACTDataRscale(incfact_data)             INCFACTDataScale(incfact_data)
  #define INCFACTDataCscale(incfact_data)             INCFACTDataScale(incfact_data)
#endif
#define INCFACTDataIwork(incfact_data)              ((incfact_data) -> iwork)
#define INCFACTDataLIwork(incfact_data)             ((incfact_data) -> l_iwork)
#define INCFACTDataRwork(incfact_data)              ((incfact_data) -> rwork)
#define INCFACTDataLRwork(incfact_data)             ((incfact_data) -> l_rwork)


#endif
