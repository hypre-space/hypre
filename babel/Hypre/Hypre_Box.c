
/******************************************************
 *
 *  File:  Hypre_Box.c
 *
 *********************************************************/

#include "Hypre_Box_Skel.h" 
#include "Hypre_Box_Data.h" 


/* *************************************************
 * Constructor
 *    Allocate Memory for private data
 *    and initialize here
 ***************************************************/
void Hypre_Box_constructor(Hypre_Box this) {
/* some relevant code segments:

typedef struct Hypre_Box_object__ *Hypre_Box;
struct Hypre_Box_object__ { ... Hypre_Box_Private d_table; ... };
typedef struct Hypre_Box_private *Hypre_Box_Private;
struct Hypre_Box_private { hypre_Box *hbox; }
*/

   this->d_table = (struct Hypre_Box_private_type*) 
     malloc(sizeof(struct Hypre_Box_private_type));
   this->d_table->hbox = hypre_BoxCreate();
} /* end constructor */

/* *************************************************
 *  Destructor
 *      deallocate memory for private data here.
 ***************************************************/
void Hypre_Box_destructor(Hypre_Box this) {
   /* JFP: Deallocates memory.
      Delete the Hypre object this object refers to, then delete
      this object's data table. */

   struct Hypre_Box_private_type *Bp = this->d_table;
   hypre_Box * B = Bp->hbox;
   hypre_BoxDestroy( B );
   free(this->d_table);
} /* end destructor */

/* ********************************************************
 * impl_Hypre_BoxNew
 **********************************************************/
void  impl_Hypre_Box_New
(Hypre_Box this, array1int lower, array1int upper, int dimension) {
/* JFP: This function initializes the data in a box. */

   int i;
   struct Hypre_Box_object_ BO = *this;
   Hypre_Box_Private BP = BO.d_table;
   struct Hypre_Box_private_type *Bp = BP;
   hypre_Box *B = Bp->hbox;

   Bp->dimension = dimension;
   for ( i=0; i<dimension; ++i ) {
      B->imin[i] = lower.data[i];
      B->imax[i] = upper.data[i];
   };

} /* end impl_Hypre_BoxNew */

/* ********************************************************
 * impl_Hypre_BoxConstructor
 **********************************************************/
Hypre_Box  impl_Hypre_Box_Constructor
(array1int lower, array1int upper, int dimension)
{
   /* declared static; just combines the new and New functions */
   Hypre_Box box = Hypre_Box_new();
   Hypre_Box_New( box, lower, upper, dimension );
   return box;
} /* end impl_Hypre_BoxConstructor */

/* ********************************************************
 * impl_Hypre_BoxSetup
 **********************************************************/
void  impl_Hypre_Box_Setup(Hypre_Box this) {
   /* nothing to do; provided for consistency with other Hypre classes */
} /* end impl_Hypre_BoxSetup */

/* ********************************************************
 * impl_Hypre_Boxprint
 **********************************************************/
void  impl_Hypre_Box_print(Hypre_Box this) {
/* JFP: This function prints, to stdout, data about the box. */

/* some relevant code segments:

typedef struct Hypre_Box_object__ *Hypre_Box;
struct Hypre_Box_object__ { ... Hypre_Box_Private d_table; ... };
typedef struct Hypre_Box_private *Hypre_Box_Private;
struct Hypre_Box_private { hypre_Box *hbox; }
*/
   int i;
   struct Hypre_Box_object_ BO = *this;
   Hypre_Box_Private BP = BO.d_table;
   struct Hypre_Box_private_type *Bp = BP;
   hypre_Box *B = Bp->hbox;

   printf( "Box imin[i],imax[i] = " );
   for ( i=0; i<Bp->dimension; ++i )
      printf( "%i,%i  ", B->imin[i], B->imax[i] );
   printf( "\n" );
} /* end impl_Hypre_Boxprint */

