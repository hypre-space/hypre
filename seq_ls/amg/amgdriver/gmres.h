/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Header for GMRES
 *
 *****************************************************************************/

#ifndef _GMRES_HEADER
#define _GMRES_HEADER


/*--------------------------------------------------------------------------
 * SPGMRPData
 *--------------------------------------------------------------------------*/

typedef struct
{
   int    (*precond)();
   void    *precond_data;

   Vector  *s;
   Vector  *r;

} SPGMRPData;

/*--------------------------------------------------------------------------
 * GMRESData
 *--------------------------------------------------------------------------*/

typedef struct
{
   int        max_krylov;
   int        max_restarts;

   void      *A_data;
   void      *P_data;
   SpgmrMem   spgmr_mem;

   char    *log_file_name;

} GMRESData;

/*--------------------------------------------------------------------------
 * Accessor functions for the GMRESData structure
 *--------------------------------------------------------------------------*/

#define GMRESDataMaxKrylov(gmres_data)    ((gmres_data) -> max_krylov)
#define GMRESDataMaxRestarts(gmres_data)  ((gmres_data) -> max_restarts)

#define GMRESDataAData(gmres_data)        ((gmres_data) -> A_data)
#define GMRESDataPData(gmres_data)        ((gmres_data) -> P_data)
#define GMRESDataSpgmrMem(gmres_data)     ((gmres_data) -> spgmr_mem)

#define GMRESDataLogFileName(gmres_data)  ((gmres_data) -> log_file_name)


#endif
