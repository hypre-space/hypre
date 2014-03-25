#
# File:          SStructVariable.py
# Symbol:        bHYPRE.SStructVariable-v1.0.0
# Symbol Type:   enumeration
# Babel Version: 1.0.4
# Description:   define enumeration constants
# 
# WARNING: Automatically generated; changes will be lost
#


"""The SStructVariable enumerated type.

An enumerated type that supports cell centered, node
centered, face centered, and edge centered variables.  Face
centered variables are split into x-face, y-face, and z-face
variables, and edge centered variables are split into x-edge,
y-edge, and z-edge variables.  The edge centered variable
types are only used in 3D.  In 2D, edge centered variables
are handled by the face centered types.

Variables are referenced relative to an abstract (cell centered)
index in the following way:
\\begin{itemize}
\\item cell centered variables are aligned with the index;
\\item node centered variables are aligned with the cell corner
at relative index (1/2, 1/2, 1/2);
\\item x-face, y-face, and z-face centered variables are aligned
with the faces at relative indexes (1/2, 0, 0), (0, 1/2, 0),
and (0, 0, 1/2), respectively;
\\item x-edge, y-edge, and z-edge centered variables are aligned
with the edges at relative indexes (0, 1/2, 1/2), (1/2, 0, 1/2),
and (1/2, 1/2, 0), respectively.
\\end{itemize}

The supported identifiers are:
\\begin{itemize}
\\item {\\tt HYPRE\\_SSTRUCT\\_VARIABLE\\_CELL}
\\item {\\tt HYPRE\\_SSTRUCT\\_VARIABLE\\_NODE}
\\item {\\tt HYPRE\\_SSTRUCT\\_VARIABLE\\_XFACE}
\\item {\\tt HYPRE\\_SSTRUCT\\_VARIABLE\\_YFACE}
\\item {\\tt HYPRE\\_SSTRUCT\\_VARIABLE\\_ZFACE}
\\item {\\tt HYPRE\\_SSTRUCT\\_VARIABLE\\_XEDGE}
\\item {\\tt HYPRE\\_SSTRUCT\\_VARIABLE\\_YEDGE}
\\item {\\tt HYPRE\\_SSTRUCT\\_VARIABLE\\_ZEDGE}
\\end{itemize}

NOTE: Although variables are referenced relative to a unique
abstract cell-centered index, some variables are associated
with multiple grid cells.  For example, node centered
variables in 3D are associated with 8 cells (away from
boundaries).  Although grid cells are distributed uniquely to
different processes, variables may be owned by multiple
processes because they may be associated with multiple cells.
"""

UNDEFINED = -1

CELL = 0

NODE = 1

XFACE = 2

YFACE = 3

ZFACE = 4

XEDGE = 5

YEDGE = 6

ZEDGE = 7

