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
 * Header for PCG
 *
 *****************************************************************************/

#ifndef _PCG_HEADER
#define _PCG_HEADER


/*--------------------------------------------------------------------------
 * PCGData
 *--------------------------------------------------------------------------*/

typedef struct
{
   int      max_iter;
   int      two_norm;

   Matrix  *A;
   Vector  *p;
   Vector  *s;
   Vector  *r;

   void   (*precond)();
   void    *precond_data;

   char    *log_file_name;

} PCGData;

/*--------------------------------------------------------------------------
 * Accessor functions for the PCGData structure
 *--------------------------------------------------------------------------*/

#define PCGDataMaxIter(pcg_data)      ((pcg_data) -> max_iter)
#define PCGDataTwoNorm(pcg_data)      ((pcg_data) -> two_norm)

#define PCGDataA(pcg_data)            ((pcg_data) -> A)
#define PCGDataP(pcg_data)            ((pcg_data) -> p)
#define PCGDataS(pcg_data)            ((pcg_data) -> s)
#define PCGDataR(pcg_data)            ((pcg_data) -> r)

#define PCGDataPrecond(pcg_data)      ((pcg_data) -> precond)
#define PCGDataPrecondData(pcg_data)  ((pcg_data) -> precond_data)

#define PCGDataLogFileName(pcg_data)  ((pcg_data) -> log_file_name)


#endif
