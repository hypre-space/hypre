/*#*****************************************************
#
#	File:  Hypre_Box.c
#
#********************************************************/

#include "Hypre_Box_Skel.h" 

hypre_Box *hypre_BoxCreate(void );
/* ... without this line, compilers think hypre_BoxCreate returns int */

/*#************************************************
#	Constructor
#**************************************************/

void Hypre_Box_constructor(Hypre_Box this) {

/* JFP: Allocates Memory */

/* some relevant code segments:

typedef struct Hypre_Box_object__ *Hypre_Box;
struct Hypre_Box_object__ { ... Hypre_Box_Private d_table; ... };
typedef struct Hypre_Box_private *Hypre_Box_Private;
struct Hypre_Box_private { hypre_Box *hbox; }
*/

   struct Hypre_Box_private * HBp;
   HBp = (struct Hypre_Box_private *)
      ( malloc(sizeof(struct Hypre_Box_private)) );
   this->d_table = (Hypre_Box_Private) HBp;
   
   this->d_table->hbox = hypre_BoxCreate();

}


/*#************************************************
#	Destructor
#**************************************************/

void Hypre_Box_destructor(Hypre_Box this) {

   /* JFP: Deallocates memory.
      Delete the Hypre object this object refers to, then delete
      this object's data table. */

   Hypre_Box_Private BP = this->d_table;
   struct Hypre_Box_private *Bp = BP;
   hypre_Box *B = Bp->hbox;

   hypre_BoxDestroy( B );

   free(this->d_table);
}

int  impl__Hypre_Box_NewBox(Hypre_Box this, array1int lower, array1int upper, int dimension) {

/* JFP: This function initializes the data in a box. */

   int i;
   struct Hypre_Box_object__ BO = *this;
   Hypre_Box_Private BP = BO.d_table;
   struct Hypre_Box_private *Bp = BP;
   hypre_Box *B = Bp->hbox;

   Bp->dimension = dimension;
   for ( i=0; i<dimension; ++i ) {
      B->imin[i] = lower.data[i];
      B->imax[i] = upper.data[i];
   };

   return 0;
}

void  impl__Hypre_Box_print(Hypre_Box this) {
/* JFP: This function prints, to stdout, data about the box. */

/* some relevant code segments:

typedef struct Hypre_Box_object__ *Hypre_Box;
struct Hypre_Box_object__ { ... Hypre_Box_Private d_table; ... };
typedef struct Hypre_Box_private *Hypre_Box_Private;
struct Hypre_Box_private { hypre_Box *hbox; }
*/
   int i;
   struct Hypre_Box_object__ BO = *this;
   Hypre_Box_Private BP = BO.d_table;
   struct Hypre_Box_private *Bp = BP;
   hypre_Box *B = Bp->hbox;

   printf( "Box imin[i],imax[i] = " );
   for ( i=0; i<Bp->dimension; ++i )
      printf( "%i,%i  ", B->imin[i], B->imax[i] );
   printf( "\n" );
}


