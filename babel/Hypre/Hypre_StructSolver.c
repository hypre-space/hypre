
/******************************************************
 *
 *  File:  Hypre_StructSolver.c
 *
 *********************************************************/

#include "Hypre_StructSolver_Skel.h" 
#include "Hypre_StructSolver_Data.h" 


/* *************************************************
 * Constructor
 *    Allocate Memory for private data
 *    and initialize here
 ***************************************************/
void Hypre_StructSolver_constructor(Hypre_StructSolver this) {
   this->d_table = (struct Hypre_StructSolver_private_type *)
      malloc( sizeof( struct Hypre_StructSolver_private_type ) );

   /* d_table contains hssolver, which is copied from
      a child object - no need to allocate here */

} /* end constructor */

/* *************************************************
 *  Destructor
 *      deallocate memory for private data here.
 ***************************************************/
void Hypre_StructSolver_destructor(Hypre_StructSolver this) {
   free(this->d_table);
} /* end destructor */

