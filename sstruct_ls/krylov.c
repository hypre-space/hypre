/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 *
 *****************************************************************************/

int hypre_KrylovCopyVector( void *x, void *y );

/*--------------------------------------------------------------------------
 * hypre_KrylovIdentitySetup
 *--------------------------------------------------------------------------*/

int
hypre_KrylovIdentitySetup( void *vdata,
                           void *A,
                           void *b,
                           void *x )

{
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_KrylovIdentity
 *--------------------------------------------------------------------------*/

int
hypre_KrylovIdentity( void *vdata,
                      void *A,
                      void *b,
                      void *x )

{
   return( hypre_KrylovCopyVector(b, x) );
}

