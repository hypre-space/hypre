
/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Member functions for hypre_AuxParVector class.
 *
 *****************************************************************************/

#include "IJ_mv.h"
#include "aux_par_vector.h"

/*--------------------------------------------------------------------------
 * hypre_AuxParVectorCreate
 *--------------------------------------------------------------------------*/

int
hypre_AuxParVectorCreate( hypre_AuxParVector **aux_vector)
{
   hypre_AuxParVector  *vector;
   
   vector = hypre_CTAlloc(hypre_AuxParVector, 1);
  
   /* set defaults */
   hypre_AuxParVectorMaxOffProcElmtsSet(vector) = 0;
   hypre_AuxParVectorCurrentNumElmtsSet(vector) = 0;
   hypre_AuxParVectorMaxOffProcElmtsAdd(vector) = 0;
   hypre_AuxParVectorCurrentNumElmtsAdd(vector) = 0;
   /* stash for setting off processor values */
   hypre_AuxParVectorOffProcISet(vector) = NULL;
   hypre_AuxParVectorOffProcDataSet(vector) = NULL;
   /* stash for adding to off processor values */
   hypre_AuxParVectorOffProcIAdd(vector) = NULL;
   hypre_AuxParVectorOffProcDataAdd(vector) = NULL;


   *aux_vector = vector;
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_AuxParVectorDestroy
 *--------------------------------------------------------------------------*/

int 
hypre_AuxParVectorDestroy( hypre_AuxParVector *vector )
{
   int ierr=0;

   if (vector)
   {
      if (hypre_AuxParVectorOffProcISet(vector))
      	    hypre_TFree(hypre_AuxParVectorOffProcISet(vector));
      if (hypre_AuxParVectorOffProcDataSet(vector))
      	    hypre_TFree(hypre_AuxParVectorOffProcDataSet(vector));
      if (hypre_AuxParVectorOffProcIAdd(vector))
      	    hypre_TFree(hypre_AuxParVectorOffProcIAdd(vector));
      if (hypre_AuxParVectorOffProcDataAdd(vector))
      	    hypre_TFree(hypre_AuxParVectorOffProcDataAdd(vector));
      hypre_TFree(vector);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_AuxParVectorInitialize
 *--------------------------------------------------------------------------*/

int 
hypre_AuxParVectorInitialize( hypre_AuxParVector *vector )
{
   int max_off_proc_elmts_set = hypre_AuxParVectorMaxOffProcElmtsSet(vector);
   int max_off_proc_elmts_add = hypre_AuxParVectorMaxOffProcElmtsAdd(vector);

   /* allocate stash for setting off processor values */
   if (max_off_proc_elmts_set > 0)
   {
      hypre_AuxParVectorOffProcISet(vector) = hypre_CTAlloc(int,
		max_off_proc_elmts_set);
      hypre_AuxParVectorOffProcDataSet(vector) = hypre_CTAlloc(double,
		max_off_proc_elmts_set);
   }
   /* allocate stash for adding to off processor values */
   if (max_off_proc_elmts_add > 0)
   {
      hypre_AuxParVectorOffProcIAdd(vector) = hypre_CTAlloc(int,
		max_off_proc_elmts_add);
      hypre_AuxParVectorOffProcDataAdd(vector) = hypre_CTAlloc(double,
		max_off_proc_elmts_add);
   }

   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_AuxParVectorSetMaxOffProcElmtsSet
 *--------------------------------------------------------------------------*/

int 
hypre_AuxParVectorSetMaxOffPRocElmtsSet( hypre_AuxParVector *vector,
					    int max_off_proc_elmts_set )
{
   int ierr = 0;
   hypre_AuxParVectorMaxOffProcElmtsSet(vector) = max_off_proc_elmts_set;
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_AuxParVectorSetMaxOffProcElmtsAdd
 *--------------------------------------------------------------------------*/

int 
hypre_AuxParVectorSetMaxOffPRocElmtsAdd( hypre_AuxParVector *vector,
					    int max_off_proc_elmts_add )
{
   int ierr = 0;
   hypre_AuxParVectorMaxOffProcElmtsAdd(vector) = max_off_proc_elmts_add;
   return ierr;
}
