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
 * Header info for the zzz_SBox ("Stride Box") structures
 *
 *****************************************************************************/

#ifndef zzz_SBOX_HEADER
#define zzz_SBOX_HEADER


/*--------------------------------------------------------------------------
 * zzz_SBox:
 *   Structure describing a strided cartesian region of some index space.
 *--------------------------------------------------------------------------*/

typedef struct
{
   zzz_Box    *box;
   zzz_Index  *stride;       /* Striding factors */

} zzz_SBox;

/*--------------------------------------------------------------------------
 * zzz_SBoxArray:
 *   An array of sboxes.
 *--------------------------------------------------------------------------*/

typedef struct
{
   zzz_SBox  **sboxes;       /* Array of pointers to sboxes */
   int         size;         /* Size of sbox array */

} zzz_SBoxArray;

#define zzz_SBoxArrayBlocksize zzz_BoxArrayBlocksize

/*--------------------------------------------------------------------------
 * zzz_SBoxArrayArray:
 *   An array of sbox arrays.
 *--------------------------------------------------------------------------*/

typedef struct
{
   zzz_SBoxArray  **sbox_arrays;   /* Array of pointers to sbox arrays */
   int              size;          /* Size of sbox array array */

} zzz_SBoxArrayArray;


/*--------------------------------------------------------------------------
 * Accessor macros: zzz_SBox
 *--------------------------------------------------------------------------*/

#define zzz_SBoxBox(sbox)         ((sbox) -> box)
#define zzz_SBoxStride(sbox)      ((sbox) -> stride)
				        
#define zzz_SBoxIMin(sbox)        zzz_BoxIMin(zzz_SBoxBox(sbox))
#define zzz_SBoxIMax(sbox)        zzz_BoxIMax(zzz_SBoxBox(sbox))
				        
#define zzz_SBoxIMinD(sbox, d)    zzz_BoxIMinD(zzz_SBoxBox(sbox), d)
#define zzz_SBoxIMaxD(sbox, d)    zzz_BoxIMaxD(zzz_SBoxBox(sbox), d)
#define zzz_SBoxStrideD(sbox, d)  zzz_IndexD(zzz_SBoxStride(sbox), d)
#define zzz_SBoxSizeD(sbox, d) \
((zzz_BoxSizeD(zzz_SBoxBox(sbox), d) - 1) / zzz_SBoxStrideD(sbox, d) + 1)
				        
#define zzz_SBoxIMinX(sbox)       zzz_SBoxIMinD(sbox, 0)
#define zzz_SBoxIMinY(sbox)       zzz_SBoxIMinD(sbox, 1)
#define zzz_SBoxIMinZ(sbox)       zzz_SBoxIMinD(sbox, 2)

#define zzz_SBoxIMaxX(sbox)       zzz_SBoxIMaxD(sbox, 0)
#define zzz_SBoxIMaxY(sbox)       zzz_SBoxIMaxD(sbox, 1)
#define zzz_SBoxIMaxZ(sbox)       zzz_SBoxIMaxD(sbox, 2)

#define zzz_SBoxStrideX(sbox)     zzz_SBoxStrideD(sbox, 0)
#define zzz_SBoxStrideY(sbox)     zzz_SBoxStrideD(sbox, 1)
#define zzz_SBoxStrideZ(sbox)     zzz_SBoxStrideD(sbox, 2)

#define zzz_SBoxSizeX(sbox)       zzz_SBoxSizeD(sbox, 0)
#define zzz_SBoxSizeY(sbox)       zzz_SBoxSizeD(sbox, 1)
#define zzz_SBoxSizeZ(sbox)       zzz_SBoxSizeD(sbox, 2)

#define zzz_SBoxVolume(sbox) \
(zzz_SBoxSizeX(sbox) * zzz_SBoxSizeY(sbox) * zzz_SBoxSizeZ(sbox))

#define zzz_GetSBoxSize(sbox, size) \
{\
   zzz_IndexX(size) = zzz_SBoxSizeX(sbox);\
   zzz_IndexY(size) = zzz_SBoxSizeY(sbox);\
   zzz_IndexZ(size) = zzz_SBoxSizeZ(sbox);\
}

/*--------------------------------------------------------------------------
 * Accessor macros: zzz_SBoxArray
 *--------------------------------------------------------------------------*/

#define zzz_SBoxArraySBoxes(sbox_array)  ((sbox_array) -> sboxes)
#define zzz_SBoxArraySBox(sbox_array, i) ((sbox_array) -> sboxes[(i)])
#define zzz_SBoxArraySize(sbox_array)    ((sbox_array) -> size)

/*--------------------------------------------------------------------------
 * Accessor macros: zzz_SBoxArrayArray
 *--------------------------------------------------------------------------*/

#define zzz_SBoxArrayArraySBoxArrays(sbox_array_array) \
((sbox_array_array) -> sbox_arrays)
#define zzz_SBoxArrayArraySBoxArray(sbox_array_array, i) \
((sbox_array_array) -> sbox_arrays[(i)])
#define zzz_SBoxArrayArraySize(sbox_array_array) \
((sbox_array_array) -> size)

/*--------------------------------------------------------------------------
 * Looping macros:
 *--------------------------------------------------------------------------*/

#define zzz_ForSBoxI(i, sbox_array) \
for (i = 0; i < zzz_SBoxArraySize(sbox_array); i++)

#define zzz_ForSBoxArrayI(i, sbox_array_array) \
for (i = 0; i < zzz_SBoxArrayArraySize(sbox_array_array); i++)


#endif
