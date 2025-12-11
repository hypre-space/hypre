!     Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
!     HYPRE Project Developers. See the top-level COPYRIGHT file for details.
!
!     SPDX-License-Identifier: (Apache-2.0 OR MIT)

!****************************************************************************
! HYPRE_SStruct_ls fortran interface routines
!****************************************************************************


!****************************************************************************
!                HYPRE_SStructBiCGSTAB routines
!****************************************************************************

!--------------------------------------------------------------------------
! HYPRE_SStructBiCGSTABCreate
!--------------------------------------------------------------------------
      subroutine fhypre_sstructbicgstabcreate(fcomm, fsolver)

      integer ierr
      integer*8 fcomm
      integer*8 fsolver

      call HYPRE_SStructBiCGSTABCreate(fcomm, fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructbicgstabcreate error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructBiCGSTABDestroy
!--------------------------------------------------------------------------
      subroutine fhypre_sstructbicgstabdestroy(fsolver)

      integer ierr
      integer*8 fsolver

       call HYPRE_SStructBiCGSTABDestroy(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructbicgstabdestroy error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructBiCGSTABSetup
!--------------------------------------------------------------------------
      subroutine fhypre_sstructbicgstabsetup(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

       call HYPRE_SStructBiCGSTABSetup(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructbicgstabsetup error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructBiCGSTABSolve
!--------------------------------------------------------------------------
      subroutine fhypre_sstructbicgstabsolve(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_SStructBiCGSTABSolve(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructbicgstabsolve error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructBiCGSTABSetTol
!--------------------------------------------------------------------------
      subroutine fhypre_sstructbicgstabsettol(fsolver, ftol)

      integer ierr
      integer*8 fsolver
      double precision ftol

      call HYPRE_SStructBiCGSTABSetTol(fsolver, ftol, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructbicgstabsettol error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructBiCGSTABSetMinIter
!--------------------------------------------------------------------------
      subroutine fhypre_sstructbicgstabsetminite(fsolver, fmin_iter)

      integer ierr
      integer fmin_iter
      integer*8 fsolver

      call HYPRE_SStructBiCGSTABSetMinIter(fsolver, fmin_iter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructbicgstabsetminiter error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructBiCGSTABSetMaxIter
!--------------------------------------------------------------------------
      subroutine fhypre_sstructbicgstabsetmaxite(fsolver, fmax_iter)

      integer ierr
      integer fmax_iter
      integer*8 fsolver

      call HYPRE_SStructBiCGSTABSetMaxIter(fsolver, fmax_iter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructbicgstabsetmaxiter error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructBiCGSTABSetStopCrit
!--------------------------------------------------------------------------
      subroutine fhypre_sstructbicgstabsetstopcr(fsolver, fstop_crit)

      integer ierr
      integer fstop_crit
      integer*8 fsolver

      call HYPRE_SStructBiCGSTABSetStopCri(fsolver, fstop_crit, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructbicgstabsetstopcrit error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructBiCGSTABSetPrecond
!--------------------------------------------------------------------------
      subroutine fhypre_sstructbicgstabsetprecon(fsolver, fprecond_id,
     1                                            fprecond)

      integer ierr
      integer fprecond_id
      integer*8 fsolver
      integer*8 fprecond

      call HYPRE_SStructBiCGSTABSetPrecond(fsolver, fprecond_id,
     1                                     fprecond, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructbicgstabsetprecond error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructBiCGSTABSetLogging
!--------------------------------------------------------------------------
      subroutine fhypre_sstructbicgstabsetloggin(fsolver, flog)

      integer ierr
      integer flog
      integer*8 fsolver

      call HYPRE_SStructBiCGSTABSetLogging(fsolver, flog, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructbicgstabsetlogging error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructBiCGSTABSetPrintLevel
!--------------------------------------------------------------------------
      subroutine fhypre_sstructbicgstabsetprintl(fsolver, fprint)

      integer ierr
      integer fprint
      integer*8 fsolver

      call HYPRE_SStructBiCGSTABSetPrintLe(fsolver, fprint, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructbicgstabsetprintlevel error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructBiCGSTABGetNumIterations
!--------------------------------------------------------------------------
      subroutine fhypre_sstructbicgstabgetnumite(fsolver, fnumiter)

      integer ierr
      integer fnumiter
      integer*8 fsolver

      call HYPRE_SStructBiCGSTABGetNumIter(fsolver, fnumiter,
     1                                     ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructbicgstabgetnumiterations error = ',
     1                                          ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructBiCGSTABGetFinalRelativeResidualNorm
!--------------------------------------------------------------------------
      subroutine fhypre_sstructbicgstabgetfinalr(fsolver, fnorm)

      integer ierr
      integer*8 fsolver
      double precision fnorm

      call HYPRE_SStructBiCGSTABGetFinalRe(fsolver, fnorm, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructbicgstabgetfinalrelative error = ',
     1                                                             ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructBiCGSTABGetResidual
!--------------------------------------------------------------------------
      subroutine fhypre_sstructbicgstabgetresidu(fsolver, fresidual)

      integer ierr
      integer*8 fsolver
      integer*8 fresidual

      call HYPRE_SStructBiCGSTABGetResidua(fsolver, fresidual, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructbicgstabgetresidual error = ', ierr
      endif

      return
      end





!****************************************************************************
!                HYPRE_SStructGMRES routines
!****************************************************************************

!--------------------------------------------------------------------------
! HYPRE_SStructGMRESCreate
!--------------------------------------------------------------------------
      subroutine fhypre_sstructgmrescreate(fcomm, fsolver)

      integer ierr
      integer*8 fcomm
      integer*8 fsolver

      call HYPRE_SStructGMRESCreate(fcomm, fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructgmrescreate error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructGMRESDestroy
!--------------------------------------------------------------------------
      subroutine fhypre_sstructgmresdestroy(fsolver)

      integer ierr
      integer*8 fsolver

      call HYPRE_SStructGMRESDestroy(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructgmresdestroy error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructGMRESSetup
!--------------------------------------------------------------------------
      subroutine fhypre_sstructgmressetup(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_SStructGMRESSetup(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructgmressetup error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructGMRESSolve
!--------------------------------------------------------------------------
      subroutine fhypre_sstructgmressolve(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_SStructGMRESSolve(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructgmressolve error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructGMRESSetKDim
!--------------------------------------------------------------------------
      subroutine fhypre_sstructgmressetkdim(fsolver, fkdim)

      integer ierr
      integer fkdim
      integer*8 fsolver

      call HYPRE_SStructGMRESSetKDim(fsolver, fkdim, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructgmressetkdim error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructGMRESSetTol
!--------------------------------------------------------------------------
      subroutine fhypre_sstructgmressettol(fsolver, ftol)

      integer ierr
      integer*8 fsolver
      double precision ftol

      call HYPRE_SStructGMRESSetTol(fsolver, ftol, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructgmressettol error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructGMRESSetMinIter
!--------------------------------------------------------------------------
      subroutine fhypre_sstructgmressetminiter(fsolver, fmin_iter)

      integer ierr
      integer fmin_iter
      integer*8 fsolver

      call HYPRE_SStructGMRESSetMinIter(fsolver, fmin_iter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructgmressetminiter error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructGMRESSetMaxIter
!--------------------------------------------------------------------------
      subroutine fhypre_sstructgmressetmaxiter(fsolver, fmax_iter)

      integer ierr
      integer fmax_iter
      integer*8 fsolver

      call HYPRE_SStructGMRESSetMaxIter(fsolver, fmax_iter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructgmressetmaxiter error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructGMRESSetStopCrit
!--------------------------------------------------------------------------
      subroutine fhypre_sstructgmressetstopcrit(fsolver, fstop_crit)

      integer ierr
      integer fstop_crit
      integer*8 fsolver

      call HYPRE_SStructGMRESSetStopCrit(fsolver, fstop_crit, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructgmressetstopcrit error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructGMRESSetPrecond
!--------------------------------------------------------------------------
      subroutine fhypre_sstructgmressetprecond(fsolver, fprecond_id,
     1                                         fprecond)

      integer ierr
      integer fprecond_id
      integer*8 fsolver
      integer*8 fprecond

      call HYPRE_SStructGMRESSetPrecond(fsolver, fprecond_id, fprecond,
     1                                  ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructgmressetprecond error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructGMRESSetLogging
!--------------------------------------------------------------------------
      subroutine fhypre_sstructgmressetlogging(fsolver, flog)

      integer ierr
      integer flog
      integer*8 fsolver

      call HYPRE_SStructGMRESSetLogging(fsolver, flog, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructgmressetlogging error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructGMRESSetPrintLevel
!--------------------------------------------------------------------------
      subroutine fhypre_sstructgmressetprintleve(fsolver, flevel)

      integer ierr
      integer flevel
      integer*8 fsolver

      call HYPRE_SStructGMRESSetPrintLevel(fsolver, flevel, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructgmressetprintlevel error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructGMRESGetNumIterations
!--------------------------------------------------------------------------
      subroutine fhypre_sstructgmresgetnumiterat(fsolver, fnumiter)

      integer ierr
      integer fnumiter
      integer*8 fsolver

      call HYPRE_SStructGMRESGetNumIterati(fsolver, fnumiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructgmresgetnumiteration error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructGMRESGetFinalRelativeResidualNorm
!--------------------------------------------------------------------------
      subroutine fhypre_sstructgmresgetfinalrela(fsolver, fnorm)

      integer ierr
      integer*8 fsolver
      double precision fnorm

      call HYPRE_SStructGMRESGetFinalRelat(fsolver, fnorm, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructgmresgetfinalrelative error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructGMRESGetResidual
!--------------------------------------------------------------------------
      subroutine fhypre_sstructgmresgetresidual(fsolver, fresidual)

      integer ierr
      integer*8 fsolver
      integer*8 fresidual

      call HYPRE_SStructGMRESGetResidual(fsolver, fresidual, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructgmresgetresidual error = ', ierr
      endif

      return
      end





!****************************************************************************
!                HYPRE_SStructInterpreter routines
!****************************************************************************

!--------------------------------------------------------------------------
! HYPRE_SStructPVectorSetRandomValues
!--------------------------------------------------------------------------
      subroutine fhypre_sstructpvectorsetrandomv(fsolver, fseed)

      integer ierr
      integer*8 fsolver
      integer*8 fseed

      call HYPRE_SStructPVectorSetRandomVa(fsolver, fseed, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructpvectorsetrandomvalues error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructVectorSetRandomValues
!--------------------------------------------------------------------------
      subroutine fhypre_sstructvectorsetrandomva(fsolver, fseed)

      integer ierr
      integer*8 fsolver
      integer*8 fseed

      call HYPRE_SStructVectorSetRandomVal(fsolver, fseed, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructvectorsetrandomvalues error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructSetRandomValues
!--------------------------------------------------------------------------
      subroutine fhypre_sstructsetrandomvalues(fsolver, fseed)

      integer ierr
      integer*8 fsolver
      integer*8 fseed

      call HYPRE_SStructSetRandomValues(fsolver, fseed, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsetrandomvalues error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructSetupInterpreter
!--------------------------------------------------------------------------
      subroutine fhypre_sstructsetupinterpreter(fsolver)

      integer ierr
      integer*8 fsolver

      call HYPRE_SStructSetupInterpreter(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsetupinterpreter error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructSetupMatvec
!--------------------------------------------------------------------------
      subroutine fhypre_sstructsetupmatvec(fsolver)

      integer ierr
      integer*8 fsolver

      call HYPRE_SStructSetupMatvec(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsetupmatvec error = ', ierr
      endif

      return
      end




!****************************************************************************
!                HYPRE_SStructPCG routines
!****************************************************************************

!--------------------------------------------------------------------------
! HYPRE_SStructPCGCreate
!--------------------------------------------------------------------------
      subroutine fhypre_sstructpcgcreate(fcomm, fsolver)

      integer ierr
      integer*8 fcomm
      integer*8 fsolver

      call HYPRE_SStructPCGCreate(fcomm, fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructpcgcreate error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructPCGDestroy
!--------------------------------------------------------------------------
      subroutine fhypre_sstructpcgdestroy(fsolver)

      integer ierr
      integer*8 fsolver

      call HYPRE_SStructPCGDestroy(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructpcgdestroy error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructPCGSetup
!--------------------------------------------------------------------------
      subroutine fhypre_sstructpcgsetup(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_SStructPCGSetup(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructpcgsetup error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructPCGSolve
!--------------------------------------------------------------------------
      subroutine fhypre_sstructpcgsolve(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_SStructPCGSolve(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructpcgsolve error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructPCGSetTol
!--------------------------------------------------------------------------
      subroutine fhypre_sstructpcgsettol(fsolver, ftol)

      integer ierr
      integer*8 fsolver
      double precision ftol

      call HYPRE_SStructPCGSetTol(fsolver, ftol, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructpcgsettol error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructPCGSetMaxIter
!--------------------------------------------------------------------------
      subroutine fhypre_sstructpcgsetmaxiter(fsolver, fmax_iter)

      integer ierr
      integer fmax_iter
      integer*8 fsolver

      call HYPRE_SStructPCGSetMaxIter(fsolver, fmax_iter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructpcgsetmaxiter error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructPCGSetTwoNorm
!--------------------------------------------------------------------------
      subroutine fhypre_sstructpcgsettwonorm(fsolver, ftwo_norm)

      integer ierr
      integer ftwo_norm
      integer*8 fsolver

      call HYPRE_SStructPCGSetTwoNorm(fsolver, ftwo_norm, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructpcgsettwonorm error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructPCGSetRelChange
!--------------------------------------------------------------------------
      subroutine fhypre_sstructpcgsetrelchange(fsolver, frel_change)

      integer ierr
      integer frel_change
      integer*8 fsolver

      call HYPRE_SStructPCGSetRelChange(fsolver, frel_change, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructpcgsetrelchange error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructPCGSetPrecond
!--------------------------------------------------------------------------
      subroutine fhypre_sstructpcgsetprecond(fsolver, fprecond_id,
     1                                       fprecond)

      integer ierr
      integer fprecond_id
      integer*8 fsolver
      integer*8 fprecond

      call HYPRE_SStructPCGSetPrecond(fsolver, fprecond_id, fprecond,
     1                                ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructpcgsetprecond error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructPCGSetLogging
!--------------------------------------------------------------------------
      subroutine fhypre_sstructpcgsetlogging(fsolver, flog)

      integer ierr
      integer flog
      integer*8 fsolver

      call HYPRE_SStructPCGSetLogging(fsolver, flog, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructpcgsetlogging error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructPCGSetPrintLevel
!--------------------------------------------------------------------------
      subroutine fhypre_sstructpcgsetprintlevel(fsolver, flevel)

      integer ierr
      integer flevel
      integer*8 fsolver

      call HYPRE_SStructPCGSetPrintLevel(fsolver, flevel, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructpcgsetprintlevel error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructPCGGetNumIterations
!--------------------------------------------------------------------------
      subroutine fhypre_sstructpcggetnumiteratio(fsolver, fnumiter)

      integer ierr
      integer fnumiter
      integer*8 fsolver

      call HYPRE_SStructPCGGetNumIteration(fsolver, fnumiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructpcggetnumiteration error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructPCGGetFinalRelativeResidualNorm
!--------------------------------------------------------------------------
      subroutine fhypre_sstructpcggetfinalrelati(fsolver, fnorm)

      integer ierr
      integer*8 fsolver
      double precision fnorm

      call HYPRE_SStructPCGGetFinalRelativ(fsolver, fnorm, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructpcggetfinalrelative error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructPCGGetResidual
!--------------------------------------------------------------------------
      subroutine fhypre_sstructpcggetresidual(fsolver, fresidual)

      integer ierr
      integer*8 fsolver
      integer*8 fresidual

      call HYPRE_SStructPCGGetResidual(fsolver, fresidual, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructpcggetresidual error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructDiagScaleSetup
!--------------------------------------------------------------------------
      subroutine fhypre_sstructdiagscalesetup(fsolver, fA, fy, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fy
      integer*8 fx

      call HYPRE_SStructDiagScaleSetup(fsolver, fA, fy, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructdiagscalesetup error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructDiagScale
!--------------------------------------------------------------------------
      subroutine fhypre_sstructdiagscale(fsolver, fA, fy, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fy
      integer*8 fx

      call HYPRE_SStructDiagScale(fsolver, fA, fy, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructdiagscale error = ', ierr
      endif

      return
      end





!****************************************************************************
!                HYPRE_SStructSplit routines
!****************************************************************************

!--------------------------------------------------------------------------
! HYPRE_SStructSplitCreate
!--------------------------------------------------------------------------
      subroutine fhypre_sstructsplitcreate(fcomm, fsolver)

      integer ierr
      integer*8 fcomm
      integer*8 fsolver

      call HYPRE_SStructSplitCreate(fcomm, fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsplitcreate error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructSplitDestroy
!--------------------------------------------------------------------------
      subroutine fhypre_sstructsplitdestroy(fsolver)

      integer ierr
      integer*8 fsolver

      call HYPRE_SStructSplitDestroy(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsplitdestroy error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructSplitSetup
!--------------------------------------------------------------------------
      subroutine fhypre_sstructsplitsetup(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_SStructSplitSetup(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsplitsetup error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructSplitSolve
!--------------------------------------------------------------------------
      subroutine fhypre_sstructsplitsolve(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_SStructSplitSolve(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsplitsolve error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructSplitSetTol
!--------------------------------------------------------------------------
      subroutine fhypre_sstructsplitsettol(fsolver, ftol)

      integer ierr
      integer*8 fsolver
      double precision ftol

      call HYPRE_SStructSplitSetTol(fsolver, ftol, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsplitsettol error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructSplitSetMaxIter
!--------------------------------------------------------------------------
      subroutine fhypre_sstructsplitsetmaxiter(fsolver, fmax_iter)

      integer ierr
      integer fmax_iter
      integer*8 fsolver

      call HYPRE_SStructSplitSetMaxIter(fsolver, fmax_iter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsplitsetmaxiter error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructSplitSetZeroGuess
!--------------------------------------------------------------------------
      subroutine fhypre_sstructsplitsetzeroguess(fsolver)

      integer ierr
      integer*8 fsolver

      call HYPRE_SStructSplitSetZeroGuess(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsplitsetzeroguess error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructSplitSetNonZeroGuess
!--------------------------------------------------------------------------
      subroutine fhypre_sstructsplitsetnonzerogu(fsolver)

      integer ierr
      integer*8 fsolver

      call HYPRE_SStructSplitSetNonZeroGue(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsplitsetnonzeroguess error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructSplitSetStructSolver
!--------------------------------------------------------------------------
      subroutine fhypre_sstructsplitsetstructsol(fsolver, fssolver)

      integer ierr
      integer fssolver
      integer*8 fsolver

      call HYPRE_SStructSplitSetStructSolv(fsolver, fssolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsplitsetstructsolve error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructSplitGetNumIterations
!--------------------------------------------------------------------------
      subroutine fhypre_sstructsplitgetnumiterat(fsolver, fnumiter)

      integer ierr
      integer fnumiter
      integer*8 fsolver

      call HYPRE_SStructSplitGetNumIterati(fsolver, fnumiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsplitgetnumiteration error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructSplitGetFinalRelativeResidualNorm
!--------------------------------------------------------------------------
      subroutine fhypre_sstructsplitgetfinalrela(fsolver, fnorm)

      integer ierr
      integer*8 fsolver
      double precision fnorm

      call HYPRE_SStructSplitGetFinalRelat(fsolver, fnorm, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsplitgetfinalrelative error = ', ierr
      endif

      return
      end





!****************************************************************************
!                HYPRE_SStructSYSPFMG routines
!****************************************************************************

!--------------------------------------------------------------------------
! HYPRE_SStructSysPFMGCreate
!--------------------------------------------------------------------------
      subroutine fhypre_sstructsyspfmgcreate(fcomm, fsolver)

      integer ierr
      integer*8 fcomm
      integer*8 fsolver

      call HYPRE_SStructSysPFMGCreate(fcomm, fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsyspfmgcreate error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructSysPFMGDestroy
!--------------------------------------------------------------------------
      subroutine fhypre_sstructsyspfmgdestroy(fsolver)

      integer ierr
      integer*8 fsolver

      call HYPRE_SStructSysPFMGDestroy(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsyspfmgdestroy error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructSysPFMGSetup
!--------------------------------------------------------------------------
      subroutine fhypre_sstructsyspfmgsetup(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_SStructSysPFMGSetup(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsyspfmgsetup error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructSysPFMGSolve
!--------------------------------------------------------------------------
      subroutine fhypre_sstructsyspfmgsolve(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_SStructSysPFMGSolve(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsyspfmgsolve error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructSysPFMGSetTol
!--------------------------------------------------------------------------
      subroutine fhypre_sstructsyspfmgsettol(fsolver, ftol)

      integer ierr
      integer*8 fsolver
      double precision ftol

      call HYPRE_SStructSysPFMGSetTol(fsolver, ftol, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsyspfmgsettol error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructSysPFMGSetMaxIter
!--------------------------------------------------------------------------
      subroutine fhypre_sstructsyspfmgsetmaxiter(fsolver, fmax_iter)

      integer ierr
      integer fmax_iter
      integer*8 fsolver

      call HYPRE_SStructSysPFMGSetMaxIter(fsolver, fmax_iter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsyspfmgsetmaxiter error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructSysPFMGSetRelChange
!--------------------------------------------------------------------------
      subroutine fhypre_sstructsyspfmgsetrelchang(fsolver, frel_change)

      integer ierr
      integer frel_change
      integer*8 fsolver

      call HYPRE_SStructSysPFMGSetRelChang(fsolver, frel_change, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsyspfmgsetrelchange error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructSysPFMGSetZeroGuess
!--------------------------------------------------------------------------
      subroutine fhypre_sstructsyspfmgsetzerogue(fsolver)

      integer ierr
      integer*8 fsolver

      call HYPRE_SStructSysPFMGSetZeroGues(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsyspfmgsetzeroguess error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructSysPFMGSetNonZeroGuess
!--------------------------------------------------------------------------
      subroutine fhypre_sstructsyspfmgsetnonzero(fsolver)

      integer ierr
      integer*8 fsolver

      call HYPRE_SStructSysPFMGSetNonZeroG(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsyspfmgsetnonzeroguess error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructSysPFMGSetRelaxType
!--------------------------------------------------------------------------
      subroutine fhypre_sstructsyspfmgsetrelaxty(fsolver, frelax_type)

      integer ierr
      integer frelax_type
      integer*8 fsolver

      call HYPRE_SStructSysPFMGSetRelaxTyp(fsolver, frelax_type, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsyspfmgsetrelaxtype error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructSysPFMGSetNumPreRelax
!--------------------------------------------------------------------------
      subroutine fhypre_sstructsyspfmgsetnumprer(fsolver,
     1                                            fnum_pre_relax)

      integer ierr
      integer fnum_pre_relax
      integer*8 fsolver

      call HYPRE_SStructSysPFMGSetNumPreRe(fsolver, fnum_pre_relax,
     1                                        ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsyspfmgsetnumprerelax error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructSysPFMGSetNumPostRelax
!--------------------------------------------------------------------------
      subroutine fhypre_sstructsyspfmgsetnumpost(fsolver,
     1                                            fnum_post_relax)

      integer ierr
      integer fnum_post_relax
      integer*8 fsolver

      call HYPRE_SStructSysPFMGSetNumPostR(fsolver, fnum_post_relax,
     1                                        ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsyspfmgsetnumpostrelax error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructSysPFMGSetSkipRelax
!--------------------------------------------------------------------------
      subroutine fhypre_sstructsyspfmgsetskiprel(fsolver, fskip_relax)

      integer ierr
      integer fskip_relax
      integer*8 fsolver

      call HYPRE_SStructSysPFMGSetSkipRela(fsolver, fskip_relax, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsyspfmgsetskiprelax error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructSysPFMGSetDxyz
!--------------------------------------------------------------------------
      subroutine fhypre_sstructsyspfmgsetdxyz(fsolver, fdxyz)

      integer ierr
      integer*8 fsolver
      double precision fdxyz

      call HYPRE_SStructSysPFMGSetDxyz(fsolver, fdxyz, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsyspfmgsetdxyz error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructSysPFMGSetLogging
!--------------------------------------------------------------------------
      subroutine fhypre_sstructsyspfmgsetlogging(fsolver, flog)

      integer ierr
      integer flog
      integer*8 fsolver

      call HYPRE_SStructSysPFMGSetLogging(fsolver, flog, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsyspfmgsetlogging error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructSysPFMGSetPrintLevel
!--------------------------------------------------------------------------
      subroutine fhypre_sstructsyspfmgsetprintle(fsolver, fprint_level)

      integer ierr
      integer fprint_level
      integer*8 fsolver

      call HYPRE_SStructSysPFMGSetPrintLev(fsolver, fprint_level,
     1                                       ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsyspfmgsetprintlevel error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructSysPFMGGetNumIterations
!--------------------------------------------------------------------------
      subroutine fhypre_sstructsyspfmggetnumiter(fsolver, fnumiter)

      integer ierr
      integer fnumiter
      integer*8 fsolver

      call HYPRE_SStructSysPFMGGetNumItera(fsolver, fnumiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsyspfmggetnumiteration error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! HYPRE_SStructSysPFMGGetFinalRelativeResidualNorm
!--------------------------------------------------------------------------
      subroutine fhypre_sstructsyspfmggetfinalre(fsolver, fnorm)

      integer ierr
      integer*8 fsolver
      double precision fnorm

      call HYPRE_SStructSysPFMGGetFinalRel(fsolver, fnorm, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsyspfmggetfinalrelative error = ', ierr
      endif

      return
      end
