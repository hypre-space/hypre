/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#ifndef HYPRE_SSTRUCT_MV_HEADER
#define HYPRE_SSTRUCT_MV_HEADER

#include "HYPRE_config.h"
#include "HYPRE_utilities.h"
#include "HYPRE.h"
#include "HYPRE_struct_mv.h"
#include "HYPRE_IJ_mv.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name SStruct System Interface
 *
 * This interface represents a semi-structured-grid conceptual view of
 * a linear system.
 *
 * @memo A semi-structured-grid conceptual interface
 * @author Robert D. Falgout
 **/
/*@{*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name SStruct Grids
 **/
/*@{*/

struct hypre_SStructGrid_struct;
/**
 * A grid object is constructed out of several structured ``parts''
 * and an optional unstructured ``part''.  Each structured part has
 * its own abstract index space.
 **/
typedef struct hypre_SStructGrid_struct *HYPRE_SStructGrid;

enum hypre_SStructVariable_enum
{
   HYPRE_SSTRUCT_VARIABLE_UNDEFINED = -1,
   HYPRE_SSTRUCT_VARIABLE_CELL      =  0,
   HYPRE_SSTRUCT_VARIABLE_NODE      =  1,
   HYPRE_SSTRUCT_VARIABLE_XFACE     =  2,
   HYPRE_SSTRUCT_VARIABLE_YFACE     =  3,
   HYPRE_SSTRUCT_VARIABLE_ZFACE     =  4,
   HYPRE_SSTRUCT_VARIABLE_XEDGE     =  5,
   HYPRE_SSTRUCT_VARIABLE_YEDGE     =  6,
   HYPRE_SSTRUCT_VARIABLE_ZEDGE     =  7
};
/**
 * An enumerated type that supports cell centered, node centered, face
 * centered, and edge centered variables.  Face centered variables are
 * split into x-face, y-face, and z-face variables, and edge centered
 * variables are split into x-edge, y-edge, and z-edge variables.  The
 * edge centered variable types are only used in 3D.  In 2D, edge
 * centered variables are handled by the face centered types.
 *
 * Variables are referenced relative to an abstract (cell centered)
 * index in the following way:
 * \begin{itemize}
 * \item cell centered variables are aligned with the index;
 * \item node centered variables are aligned with the cell corner
 *       at relative index (1/2, 1/2, 1/2);
 * \item x-face, y-face, and z-face centered variables are aligned
 *       with the faces at relative indexes (1/2, 0, 0), (0, 1/2, 0),
 *       and (0, 0, 1/2), respectively;
 * \item x-edge, y-edge, and z-edge centered variables are aligned
 *       with the edges at relative indexes (0, 1/2, 1/2), (1/2, 0, 1/2),
 *       and (1/2, 1/2, 0), respectively.
 * \end{itemize}
 *
 * The supported identifiers are:
 * \begin{itemize}
 * \item {\tt HYPRE\_SSTRUCT\_VARIABLE\_CELL}
 * \item {\tt HYPRE\_SSTRUCT\_VARIABLE\_NODE}
 * \item {\tt HYPRE\_SSTRUCT\_VARIABLE\_XFACE}
 * \item {\tt HYPRE\_SSTRUCT\_VARIABLE\_YFACE}
 * \item {\tt HYPRE\_SSTRUCT\_VARIABLE\_ZFACE}
 * \item {\tt HYPRE\_SSTRUCT\_VARIABLE\_XEDGE}
 * \item {\tt HYPRE\_SSTRUCT\_VARIABLE\_YEDGE}
 * \item {\tt HYPRE\_SSTRUCT\_VARIABLE\_ZEDGE}
 * \end{itemize}
 *
 * NOTE: Although variables are referenced relative to a unique
 * abstract cell-centered index, some variables are associated with
 * multiple grid cells.  For example, node centered variables in 3D
 * are associated with 8 cells (away from boundaries).  Although grid
 * cells are distributed uniquely to different processes, variables
 * may be owned by multiple processes because they may be associated
 * with multiple cells.
 **/
typedef enum hypre_SStructVariable_enum HYPRE_SStructVariable;

/**
 * Create an {\tt ndim}-dimensional grid object with {\tt nparts}
 * structured parts.
 **/
int HYPRE_SStructGridCreate(MPI_Comm           comm,
                            int                ndim,
                            int                nparts,
                            HYPRE_SStructGrid *grid);

/**
 * Destroy a grid object.  An object should be explicitly destroyed
 * using this destructor when the user's code no longer needs direct
 * access to it.  Once destroyed, the object must not be referenced
 * again.  Note that the object may not be deallocated at the
 * completion of this call, since there may be internal package
 * references to the object.  The object will then be destroyed when
 * all internal reference counts go to zero.
 **/
int HYPRE_SStructGridDestroy(HYPRE_SStructGrid grid);

/**
 * Set the extents for a box on a structured part of the grid.
 **/
int HYPRE_SStructGridSetExtents(HYPRE_SStructGrid  grid,
                                int                part,
                                int               *ilower,
                                int               *iupper);

/**
 * Describe the variables that live on a structured part of the grid.
 **/
int HYPRE_SStructGridSetVariables(HYPRE_SStructGrid      grid,
                                  int                    part,
                                  int                    nvars,
                                  HYPRE_SStructVariable *vartypes);

/**
 * Describe additional variables that live at a particular index.
 * These variables are appended to the array of variables set in
 * \Ref{HYPRE_SStructGridSetVariables}, and are referenced as such.
 **/
int HYPRE_SStructGridAddVariables(HYPRE_SStructGrid      grid,
                                  int                    part,
                                  int                   *index,
                                  int                    nvars,
                                  HYPRE_SStructVariable *vartypes);

/**
 * Describe how regions just outside of a part relate to other parts.
 * This is done a box at a time.
 *
 * The indexes {\tt ilower} and {\tt iupper} map directly to the
 * indexes {\tt nbor\_ilower} and {\tt nbor\_iupper}.  Although, it is
 * required that indexes increase from {\tt ilower} to {\tt iupper},
 * indexes may increase and/or decrease from {\tt nbor\_ilower} to
 * {\tt nbor\_iupper}.
 * 
 * The {\tt index\_map} describes the mapping of indexes 0, 1, and 2
 * on part {\tt part} to the corresponding indexes on part {\tt
 * nbor\_part}.  For example, triple (1, 2, 0) means that indexes 0,
 * 1, and 2 on part {\tt part} map to indexes 1, 2, and 0 on part {\tt
 * nbor\_part}, respectively.
 *
 * NOTE: All parts related to each other via this routine must have an
 * identical list of variables and variable types.  For example, if
 * part 0 has only two variables on it, a cell centered variable and a
 * node centered variable, and we declare part 1 to be a neighbor of
 * part 0, then part 1 must also have only two variables on it, and
 * they must be of type cell and node.
 **/
int HYPRE_SStructGridSetNeighborBox(HYPRE_SStructGrid  grid,
                                    int                part,
                                    int               *ilower,
                                    int               *iupper,
                                    int                nbor_part,
                                    int               *nbor_ilower,
                                    int               *nbor_iupper,
                                    int               *index_map);

/**
 * Add an unstructured part to the grid.  The variables in the
 * unstructured part of the grid are referenced by a global rank
 * between 0 and the total number of unstructured variables minus one.
 * Each process owns some unique consecutive range of variables,
 * defined by {\tt ilower} and {\tt iupper}.
 *
 * NOTE: This is just a placeholder.  This part of the interface
 * is not finished.
 **/
int HYPRE_SStructGridAddUnstructuredPart(HYPRE_SStructGrid grid,
                                         int               ilower,
                                         int               iupper);

/**
 * Finalize the construction of the grid before using.
 **/
int HYPRE_SStructGridAssemble(HYPRE_SStructGrid grid);

/**
 * (Optional) Set periodic for a particular part.
 **/
int HYPRE_SStructGridSetPeriodic(HYPRE_SStructGrid  grid,
                                 int                part,
                                 int               *periodic);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name SStruct Stencils
 **/
/*@{*/

struct hypre_SStructStencil_struct;
/**
 * The stencil object.
 **/
typedef struct hypre_SStructStencil_struct *HYPRE_SStructStencil;

/**
 * Create a stencil object for the specified number of spatial dimensions
 * and stencil entries.
 **/
int HYPRE_SStructStencilCreate(int                   ndim,
                               int                   size,
                               HYPRE_SStructStencil *stencil);

/**
 * Destroy a stencil object.
 **/
int HYPRE_SStructStencilDestroy(HYPRE_SStructStencil stencil);

/**
 * Set a stencil entry.
 **/
int HYPRE_SStructStencilSetEntry(HYPRE_SStructStencil  stencil,
                                 int                   entry,
                                 int                  *offset,
                                 int                   var);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name SStruct Graphs
 **/
/*@{*/

struct hypre_SStructGraph_struct;
/**
 * The graph object is used to describe the nonzero structure of a
 * matrix.
 **/
typedef struct hypre_SStructGraph_struct *HYPRE_SStructGraph;

/**
 * Create a graph object.
 **/
int HYPRE_SStructGraphCreate(MPI_Comm             comm,
                             HYPRE_SStructGrid    grid,
                             HYPRE_SStructGraph  *graph);

/**
 * Destroy a graph object.
 **/
int HYPRE_SStructGraphDestroy(HYPRE_SStructGraph graph);

/**
 * Set the stencil for a variable on a structured part of the grid.
 **/
int HYPRE_SStructGraphSetStencil(HYPRE_SStructGraph   graph,
                                 int                  part,
                                 int                  var,
                                 HYPRE_SStructStencil stencil);

/**
 * Add a non-stencil graph entry at a particular index.  This graph
 * entry is appended to the existing graph entries, and is referenced
 * as such.
 *
 * NOTE: Users are required to set graph entries on all processes that
 * own the associated variables.  This means that some data will be
 * multiply defined.
 **/
int HYPRE_SStructGraphAddEntries(HYPRE_SStructGraph   graph,
                                 int                  part,
                                 int                 *index,
                                 int                  var,
                                 int                  to_part,
                                 int                 *to_index,
                                 int                  to_var);

/**
 * Finalize the construction of the graph before using.
 **/
int HYPRE_SStructGraphAssemble(HYPRE_SStructGraph graph);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name SStruct Matrices
 **/
/*@{*/

struct hypre_SStructMatrix_struct;
/**
 * The matrix object.
 **/
typedef struct hypre_SStructMatrix_struct *HYPRE_SStructMatrix;

/**
 * Create a matrix object.
 **/
int HYPRE_SStructMatrixCreate(MPI_Comm              comm,
                              HYPRE_SStructGraph    graph,
                              HYPRE_SStructMatrix  *matrix);

/**
 * Destroy a matrix object.
 **/
int HYPRE_SStructMatrixDestroy(HYPRE_SStructMatrix matrix);

/**
 * Prepare a matrix object for setting coefficient values.
 **/
int HYPRE_SStructMatrixInitialize(HYPRE_SStructMatrix matrix);

/**
 * Set matrix coefficients index by index.
 *
 * NOTE: Users are required to set values on all processes that own
 * the associated variables.  This means that some data will be
 * multiply defined.
 *
 * NOTE: The entries in this routine must all be of the same type:
 * either stencil or non-stencil, but not both.  Also, if they are
 * stencil entries, they must all represent couplings to the same
 * variable type (there are no such restrictions for non-stencil
 * entries).
 *
 * If the matrix is complex, then {\tt values} consists of pairs of
 * doubles representing the real and imaginary parts of each complex
 * value.
 *
 * @see HYPRE_SStructMatrixSetComplex
 **/
int HYPRE_SStructMatrixSetValues(HYPRE_SStructMatrix  matrix,
                                 int                  part,
                                 int                 *index,
                                 int                  var,
                                 int                  nentries,
                                 int                 *entries,
                                 double              *values);

/**
 * Set matrix coefficients a box at a time.
 *
 * NOTE: Users are required to set values on all processes that own
 * the associated variables.  This means that some data will be
 * multiply defined.
 *
 * NOTE: The entries in this routine must all be of the same type:
 * either stencil or non-stencil, but not both.  Also, if they are
 * stencil entries, they must all represent couplings to the same
 * variable type (there are no such restrictions for non-stencil
 * entries).
 *
 * If the matrix is complex, then {\tt values} consists of pairs of
 * doubles representing the real and imaginary parts of each complex
 * value.
 *
 * @see HYPRE_SStructMatrixSetComplex
 **/
int HYPRE_SStructMatrixSetBoxValues(HYPRE_SStructMatrix  matrix,
                                    int                  part,
                                    int                 *ilower,
                                    int                 *iupper,
                                    int                  var,
                                    int                  nentries,
                                    int                 *entries,
                                    double              *values);
/**
 * Add to matrix coefficients index by index.
 *
 * NOTE: Users are required to set values on all processes that own
 * the associated variables.  This means that some data will be
 * multiply defined.
 *
 * NOTE: The entries in this routine must all be of the same type:
 * either stencil or non-stencil, but not both.  Also, if they are
 * stencil entries, they must all represent couplings to the same
 * variable type.
 *
 * If the matrix is complex, then {\tt values} consists of pairs of
 * doubles representing the real and imaginary parts of each complex
 * value.
 *
 * @see HYPRE_SStructMatrixSetComplex
 **/
int HYPRE_SStructMatrixAddToValues(HYPRE_SStructMatrix  matrix,
                                   int                  part,
                                   int                 *index,
                                   int                  var,
                                   int                  nentries,
                                   int                 *entries,
                                   double              *values);

/**
 * Add to matrix coefficients a box at a time.
 *
 * NOTE: Users are required to set values on all processes that own
 * the associated variables.  This means that some data will be
 * multiply defined.
 *
 * NOTE: The entries in this routine must all be of stencil type.
 * Also, they must all represent couplings to the same variable type.
 *
 * If the matrix is complex, then {\tt values} consists of pairs of
 * doubles representing the real and imaginary parts of each complex
 * value.
 *
 * @see HYPRE_SStructMatrixSetComplex
 **/
int HYPRE_SStructMatrixAddToBoxValues(HYPRE_SStructMatrix  matrix,
                                      int                  part,
                                      int                 *ilower,
                                      int                 *iupper,
                                      int                  var,
                                      int                  nentries,
                                      int                 *entries,
                                      double              *values);

/**
 * Finalize the construction of the matrix before using.
 **/
int HYPRE_SStructMatrixAssemble(HYPRE_SStructMatrix matrix);

/**
 * Define symmetry properties of the matrix.  By default, matrices are
 * assumed to be nonsymmetric.  Significant storage savings can be
 * made if the matrix is symmetric.
 **/
int HYPRE_SStructMatrixSetSymmetric(HYPRE_SStructMatrix  matrix,
                                    int                  symmetric);

/**
 * Set the storage type of the matrix object to be constructed.
 * Currently, {\tt type} can be either {\tt HYPRE\_SSTRUCT} (the
 * default) or {\tt HYPRE\_PARCSR}.
 *
 * @see HYPRE_SStructMatrixGetObject
 **/
int HYPRE_SStructMatrixSetObjectType(HYPRE_SStructMatrix  matrix,
                                     int                  type);

/**
 * Get a reference to the constructed matrix object.
 *
 * @see HYPRE_SStructMatrixSetObjectType
 **/
int HYPRE_SStructMatrixGetObject(HYPRE_SStructMatrix   matrix,
                                 void                **object);

/**
 * Set the matrix to be complex.
 **/
int HYPRE_SStructMatrixSetComplex(HYPRE_SStructMatrix matrix);

/**
 * Print the matrix to file.  This is mainly for debugging purposes.
 **/
int HYPRE_SStructMatrixPrint(const char          *filename,
                             HYPRE_SStructMatrix  matrix,
                             int                  all);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name SStruct Vectors
 **/
/*@{*/

struct hypre_SStructVector_struct;
/**
 * The vector object.
 **/
typedef struct hypre_SStructVector_struct *HYPRE_SStructVector;

/**
 * Create a vector object.
 **/
int HYPRE_SStructVectorCreate(MPI_Comm              comm,
                              HYPRE_SStructGrid     grid,
                              HYPRE_SStructVector  *vector);

/**
 * Destroy a vector object.
 **/
int HYPRE_SStructVectorDestroy(HYPRE_SStructVector vector);

/**
 * Prepare a vector object for setting coefficient values.
 **/
int HYPRE_SStructVectorInitialize(HYPRE_SStructVector vector);

/**
 * Set vector coefficients index by index.
 *
 * NOTE: Users are required to set values on all processes that own
 * the associated variables.  This means that some data will be
 * multiply defined.
 *
 * If the vector is complex, then {\tt value} consists of a pair of
 * doubles representing the real and imaginary parts of the complex
 * value.
 *
 * @see HYPRE_SStructVectorSetComplex
 **/
int HYPRE_SStructVectorSetValues(HYPRE_SStructVector  vector,
                                 int                  part,
                                 int                 *index,
                                 int                  var,
                                 double              *value);

/**
 * Set vector coefficients a box at a time.
 *
 * NOTE: Users are required to set values on all processes that own
 * the associated variables.  This means that some data will be
 * multiply defined.
 *
 * If the vector is complex, then {\tt values} consists of pairs of
 * doubles representing the real and imaginary parts of each complex
 * value.
 *
 * @see HYPRE_SStructVectorSetComplex
 **/
int HYPRE_SStructVectorSetBoxValues(HYPRE_SStructVector  vector,
                                    int                  part,
                                    int                 *ilower,
                                    int                 *iupper,
                                    int                  var,
                                    double              *values);
/**
 * Set vector coefficients index by index.
 *
 * NOTE: Users are required to set values on all processes that own
 * the associated variables.  This means that some data will be
 * multiply defined.
 *
 * If the vector is complex, then {\tt value} consists of a pair of
 * doubles representing the real and imaginary parts of the complex
 * value.
 *
 * @see HYPRE_SStructVectorSetComplex
 **/
int HYPRE_SStructVectorAddToValues(HYPRE_SStructVector  vector,
                                   int                  part,
                                   int                 *index,
                                   int                  var,
                                   double              *value);

/**
 * Set vector coefficients a box at a time.
 *
 * NOTE: Users are required to set values on all processes that own
 * the associated variables.  This means that some data will be
 * multiply defined.
 *
 * If the vector is complex, then {\tt values} consists of pairs of
 * doubles representing the real and imaginary parts of each complex
 * value.
 *
 * @see HYPRE_SStructVectorSetComplex
 **/
int HYPRE_SStructVectorAddToBoxValues(HYPRE_SStructVector  vector,
                                      int                  part,
                                      int                 *ilower,
                                      int                 *iupper,
                                      int                  var,
                                      double              *values);

/**
 * Finalize the construction of the vector before using.
 **/
int HYPRE_SStructVectorAssemble(HYPRE_SStructVector vector);


/**
 * Gather vector data so that efficient {\tt GetValues} can be done.
 **/
int HYPRE_SStructVectorGather(HYPRE_SStructVector vector);


/**
 * Get vector coefficients index by index.
 *
 * NOTE: Users may only get values on processes that own the
 * associated variables.
 *
 * If the vector is complex, then {\tt value} consists of a pair of
 * doubles representing the real and imaginary parts of the complex
 * value.
 *
 * @see HYPRE_SStructVectorSetComplex
 **/
int HYPRE_SStructVectorGetValues(HYPRE_SStructVector  vector,
                                 int                  part,
                                 int                 *index,
                                 int                  var,
                                 double              *value);

/**
 * Get vector coefficients a box at a time.
 *
 * NOTE: Users may only get values on processes that own the
 * associated variables.
 *
 * If the vector is complex, then {\tt values} consists of pairs of
 * doubles representing the real and imaginary parts of each complex
 * value.
 *
 * @see HYPRE_SStructVectorSetComplex
 **/
int HYPRE_SStructVectorGetBoxValues(HYPRE_SStructVector  vector,
                                    int                  part,
                                    int                 *ilower,
                                    int                 *iupper,
                                    int                  var,
                                    double              *values);

/**
 * Set the storage type of the vector object to be constructed.
 * Currently, {\tt type} can be either {\tt HYPRE\_SSTRUCT} (the
 * default) or {\tt HYPRE\_PARCSR}.
 *
 * @see HYPRE_SStructVectorGetObject
 **/
int HYPRE_SStructVectorSetObjectType(HYPRE_SStructVector  vector,
                                     int                  type);

/**
 * Get a reference to the constructed vector object.
 *
 * @see HYPRE_SStructVectorSetObjectType
 **/
int HYPRE_SStructVectorGetObject(HYPRE_SStructVector   vector,
                                 void                **object);

/**
 * Set the vector to be complex.
 **/
int HYPRE_SStructVectorSetComplex(HYPRE_SStructVector vector);

/**
 * Print the vector to file.  This is mainly for debugging purposes.
 **/
int HYPRE_SStructVectorPrint(const char          *filename,
                             HYPRE_SStructVector  vector,
                             int                  all);

/*@}*/
/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

#ifdef __cplusplus
}
#endif

#endif

