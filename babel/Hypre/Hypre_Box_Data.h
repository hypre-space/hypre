/*#*****************************************************
#
#	File:  Hypre_Box_DataMembers.h
#
#********************************************************/

#ifndef Hypre_Box_DataMembers__
#define Hypre_Box_DataMembers__

#include "struct_matrix_vector.h" /*gkk:  also defines hypre_Box in hypre/include dir ??? */
/* JFP... */
/* #include "box.h"   /* can be found in hypre's struct_matrix_vector directory */
/* can't make this work: typedef hypre_Box Hypre_Box_private; */

struct Hypre_Box_private_type
{
   hypre_Box *hbox;
   int dimension;
}
;
#endif

