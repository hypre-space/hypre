
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

typedef struct Hypre_Box_ior *Hypre_Box;
struct Hypre_Box_ior { ... Hypre_Box_Private Hypre_Box_data; ... };
typedef struct Hypre_Box_private *Hypre_Box_Private;
struct Hypre_Box_private { hypre_Box *hbox; }
*/

   this->Hypre_Box_data = (struct Hypre_Box_private_type*) 
     malloc(sizeof(struct Hypre_Box_private_type));
   this->Hypre_Box_data->hbox = hypre_BoxCreate();
} /* end constructor */

/* *************************************************
 *  Destructor
 *      deallocate memory for private data here.
 ***************************************************/
void Hypre_Box_destructor(Hypre_Box this) {
   /* JFP: Deallocates memory.
      Delete the Hypre object this object refers to, then delete
      this object's data table. */

   struct Hypre_Box_private_type *Bp = this->Hypre_Box_data;
   hypre_Box * B = Bp->hbox;
   hypre_BoxDestroy( B );
   free(this->Hypre_Box_data);
} /* end destructor */

/* ********************************************************
 * impl_Hypre_Box_Start
 **********************************************************/
int  impl_Hypre_Box_Start
(Hypre_Box this, array1int lower, array1int upper, int dimension) {
/* JFP: This function initializes the data in a box. */

   int i;
   struct Hypre_Box_ior BO = *this;
   Hypre_Box_Private BP = BO.Hypre_Box_data;
   struct Hypre_Box_private_type *Bp = BP;
   hypre_Box *B = Bp->hbox;

   Bp->dimension = dimension;
   for ( i=0; i<dimension; ++i ) {
      B->imin[i] = lower.data[i];
      B->imax[i] = upper.data[i];
   };

} /* end impl_Hypre_Box_Start */

/* ********************************************************
 * impl_Hypre_BoxConstructor
 **********************************************************/
Hypre_Box  impl_Hypre_Box_Constructor
(array1int lower, array1int upper, int dimension)
{
   /* declared static; just combines the new and Start functions */
   Hypre_Box box = Hypre_Box_New();
   Hypre_Box_Start( box, lower, upper, dimension );
   return box;
} /* end impl_Hypre_BoxConstructor */

/* ********************************************************
 * impl_Hypre_BoxSetup
 **********************************************************/
int  impl_Hypre_Box_Setup(Hypre_Box this) {
   /* nothing to do; provided for consistency with other Hypre classes */
   return 0;
} /* end impl_Hypre_BoxSetup */

/* ********************************************************
 * impl_Hypre_Boxprint
 **********************************************************/
void  impl_Hypre_Box_print(Hypre_Box this) {
/* JFP: This function prints, to stdout, data about the box. */

/* some relevant code segments:

typedef struct Hypre_Box_ior *Hypre_Box;
struct Hypre_Box_ior { ... Hypre_Box_Private Hypre_Box_data; ... };
typedef struct Hypre_Box_private *Hypre_Box_Private;
struct Hypre_Box_private { hypre_Box *hbox; }
*/
   int i;
   struct Hypre_Box_ior BO = *this;
   Hypre_Box_Private BP = BO.Hypre_Box_data;
   struct Hypre_Box_private_type *Bp = BP;
   hypre_Box *B = Bp->hbox;

   printf( "Box imin[i],imax[i] = " );
   for ( i=0; i<Bp->dimension; ++i )
      printf( "%i,%i  ", B->imin[i], B->imax[i] );
   printf( "\n" );
} /* end impl_Hypre_Boxprint */

