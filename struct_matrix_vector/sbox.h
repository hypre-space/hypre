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
 * Header info for the hypre_SBox ("Stride Box") structures
 *
 *****************************************************************************/

#ifndef hypre_SBOX_HEADER
#define hypre_SBOX_HEADER


/*--------------------------------------------------------------------------
 * hypre_SBox:
 *   Structure describing a strided cartesian region of some index space.
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_Box    *box;
   hypre_Index   stride;       /* Striding factors */

} hypre_SBox;

/*--------------------------------------------------------------------------
 * hypre_SBoxArray:
 *   An array of sboxes.
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_SBox  **sboxes;       /* Array of pointers to sboxes */
   int         size;         /* Size of sbox array */

} hypre_SBoxArray;

#define hypre_SBoxArrayBlocksize hypre_BoxArrayBlocksize

/*--------------------------------------------------------------------------
 * hypre_SBoxArrayArray:
 *   An array of sbox arrays.
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_SBoxArray  **sbox_arrays;   /* Array of pointers to sbox arrays */
   int              size;          /* Size of sbox array array */

} hypre_SBoxArrayArray;


/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SBox
 *--------------------------------------------------------------------------*/

#define hypre_SBoxBox(sbox)         ((sbox) -> box)
#define hypre_SBoxStride(sbox)      ((sbox) -> stride)
				        
#define hypre_SBoxIMin(sbox)        hypre_BoxIMin(hypre_SBoxBox(sbox))
#define hypre_SBoxIMax(sbox)        hypre_BoxIMax(hypre_SBoxBox(sbox))
				        
#define hypre_SBoxIMinD(sbox, d)    hypre_BoxIMinD(hypre_SBoxBox(sbox), d)
#define hypre_SBoxIMaxD(sbox, d)    hypre_BoxIMaxD(hypre_SBoxBox(sbox), d)
#define hypre_SBoxStrideD(sbox, d)  hypre_IndexD(hypre_SBoxStride(sbox), d)
#define hypre_SBoxSizeD(sbox, d) \
((hypre_BoxSizeD(hypre_SBoxBox(sbox), d) - 1) / hypre_SBoxStrideD(sbox, d) + 1)
				        
#define hypre_SBoxIMinX(sbox)       hypre_SBoxIMinD(sbox, 0)
#define hypre_SBoxIMinY(sbox)       hypre_SBoxIMinD(sbox, 1)
#define hypre_SBoxIMinZ(sbox)       hypre_SBoxIMinD(sbox, 2)

#define hypre_SBoxIMaxX(sbox)       hypre_SBoxIMaxD(sbox, 0)
#define hypre_SBoxIMaxY(sbox)       hypre_SBoxIMaxD(sbox, 1)
#define hypre_SBoxIMaxZ(sbox)       hypre_SBoxIMaxD(sbox, 2)

#define hypre_SBoxStrideX(sbox)     hypre_SBoxStrideD(sbox, 0)
#define hypre_SBoxStrideY(sbox)     hypre_SBoxStrideD(sbox, 1)
#define hypre_SBoxStrideZ(sbox)     hypre_SBoxStrideD(sbox, 2)

#define hypre_SBoxSizeX(sbox)       hypre_SBoxSizeD(sbox, 0)
#define hypre_SBoxSizeY(sbox)       hypre_SBoxSizeD(sbox, 1)
#define hypre_SBoxSizeZ(sbox)       hypre_SBoxSizeD(sbox, 2)

#define hypre_SBoxVolume(sbox) \
(hypre_SBoxSizeX(sbox) * hypre_SBoxSizeY(sbox) * hypre_SBoxSizeZ(sbox))

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SBoxArray
 *--------------------------------------------------------------------------*/

#define hypre_SBoxArraySBoxes(sbox_array)  ((sbox_array) -> sboxes)
#define hypre_SBoxArraySBox(sbox_array, i) ((sbox_array) -> sboxes[(i)])
#define hypre_SBoxArraySize(sbox_array)    ((sbox_array) -> size)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SBoxArrayArray
 *--------------------------------------------------------------------------*/

#define hypre_SBoxArrayArraySBoxArrays(sbox_array_array) \
((sbox_array_array) -> sbox_arrays)
#define hypre_SBoxArrayArraySBoxArray(sbox_array_array, i) \
((sbox_array_array) -> sbox_arrays[(i)])
#define hypre_SBoxArrayArraySize(sbox_array_array) \
((sbox_array_array) -> size)

/*--------------------------------------------------------------------------
 * Looping macros:
 *--------------------------------------------------------------------------*/

#define hypre_ForSBoxI(i, sbox_array) \
for (i = 0; i < hypre_SBoxArraySize(sbox_array); i++)

#define hypre_ForSBoxArrayI(i, sbox_array_array) \
for (i = 0; i < hypre_SBoxArrayArraySize(sbox_array_array); i++)


#endif
