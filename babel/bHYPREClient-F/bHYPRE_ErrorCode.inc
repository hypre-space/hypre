C       
C       File:          bHYPRE_ErrorCode.inc
C       Symbol:        bHYPRE.ErrorCode-v1.0.0
C       Symbol Type:   enumeration
C       Babel Version: 1.0.4
C       Description:   Client-side glue code for bHYPRE.ErrorCode
C       
C       WARNING: Automatically generated; changes will be lost
C       


C       
C       Symbol "bHYPRE.ErrorCode" (version 1.0.0)
C       
C       The ErrorCode enumerated type is used with methods of the ErrorHandler class.
C       


        integer*4 HYPRE_ERROR_GENERIC
        parameter (HYPRE_ERROR_GENERIC = 1)

        integer*4 HYPRE_ERROR_MEMORY
        parameter (HYPRE_ERROR_MEMORY = 2)

        integer*4 HYPRE_ERROR_ARG
        parameter (HYPRE_ERROR_ARG = 4)

        integer*4 HYPRE_ERROR_CONV
        parameter (HYPRE_ERROR_CONV = 256)

