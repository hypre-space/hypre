c****************************************************************************
c HYPRE_SStruct fortran interface routines
c****************************************************************************

c--------------------------------------------------------------------------
c HYPRE_SStructBiCGSTABCreate
c--------------------------------------------------------------------------
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


c--------------------------------------------------------------------------
c HYPRE_SStructBiCGSTABDestroy
c--------------------------------------------------------------------------
      subroutine fhypre_sstructbicgstabdestroy(fsolver)

      integer ierr
      integer*8 fsolver

       call HYPRE_SStructBiCGSTABDestroy(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructbicgstabdestroy error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c HYPRE_SStructBiCGSTABSetup
c--------------------------------------------------------------------------
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


c--------------------------------------------------------------------------
c HYPRE_SStructBiCGSTABSolve
c--------------------------------------------------------------------------
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


c--------------------------------------------------------------------------
c HYPRE_SStructBiCGSTABSetTol
c--------------------------------------------------------------------------
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


c--------------------------------------------------------------------------
c HYPRE_SStructBiCGSTABSetMinIter
c--------------------------------------------------------------------------
      subroutine fhypre_sstructbicgstabsetminiter(fsolver, fmin_iter)

      integer ierr
      integer fmin_iter
      integer*8 fsolver

      call HYPRE_SStructBiCGSTABSetMinIter(fsolver, fmin_iter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructbicgstabsetminiter error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c HYPRE_SStructBiCGSTABSetMaxIter
c--------------------------------------------------------------------------
      subroutine fhypre_sstructbicgstabsetmaxiter(fsolver, fmax_iter)

      integer ierr
      integer fmax_iter
      integer*8 fsolver

      call HYPRE_SStructBiCGSTABSetMaxIter(fsolver, fmax_iter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructbicgstabsetmaxiter error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c HYPRE_SStructBiCGSTABSetStopCrit
c--------------------------------------------------------------------------
      subroutine fhypre_sstructbicgstabsetstopcri(fsolver, fstop_crit)

      integer ierr
      integer fstop_crit
      integer*8 fsolver

      call HYPRE_SStructBiCGSTABSetStopCrit(fsolver, fstop_crit, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructbicgstabsetstopcrit error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c HYPRE_SStructBiCGSTABSetPrecond
c--------------------------------------------------------------------------
      subroutine fhypre_sstructbicgstabsetprecond(fsolver, fprecond_id,
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


c--------------------------------------------------------------------------
c HYPRE_SStructBiCGSTABSetLogging
c--------------------------------------------------------------------------
      subroutine fhypre_sstructbicgstabsetlogging(fsolver, flog)

      integer ierr
      integer flog
      integer*8 fsolver

      call HYPRE_SStructBiCGSTABSetLogging(fsolver, flog, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructbicgstabsetlogging error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c HYPRE_SStructBiCGSTABSetPrintLevel
c--------------------------------------------------------------------------
      subroutine fhypre_sstructbicgstabsetprintle(fsolver, fprint)

      integer ierr
      integer fprint
      integer*8 fsolver

      call HYPRE_SStructBiCGSTABSetPrintLevel(fsolver, fprint, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructbicgstabsetprintlevel error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c HYPRE_SStructBiCGSTABGetNumIterations
c--------------------------------------------------------------------------
      subroutine fhypre_sstructbicgstabgetnumiter(fsolver, fnumiter)

      integer ierr
      integer fnumiter
      integer*8 fsolver

      call HYPRE_SStructBiCGSTABGetNumIterations(fsolver, fnumiter, 
     1                                           ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructbicgstabgetnumiterations error = ', 
     1                                          ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c HYPRE_SStructBiCGSTABGetFinalRelativeResidualNorm
c--------------------------------------------------------------------------
      subroutine fhypre_sstructbicgstabgetfinalre(fsolver, fnorm)

      integer ierr
      integer*8 fsolver
      double precision fnorm

      call HYPRE_SStructBiCGSTABGetFinalRelativeResidualNorm(fsolver,
     1                                                      fnorm, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructbicgstabgetfinalrelative error = ',
     1                                                             ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c HYPRE_SStructBiCGSTABGetResidual
c--------------------------------------------------------------------------
      subroutine fhypre_sstructbicgstabgetresidua(fsolver, fresidual)

      integer ierr
      integer*8 fsolver
      integer*8 fresidual

      call HYPRE_SStructBiCGSTABGetResidual(fsolver, fresidual, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructbicgstabgetresidual error = ', ierr
      endif

      return
      end



c--------------------------------------------------------------------------
c HYPRE_SStructGMRESCreate
c--------------------------------------------------------------------------
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


c--------------------------------------------------------------------------
c HYPRE_SStructGMRESDestroy
c--------------------------------------------------------------------------
      subroutine fhypre_sstructgmresdestroy(fsolver)

      integer ierr
      integer*8 fsolver

      call HYPRE_SStructGMRESDestroy(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructgmresdestroy error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c HYPRE_SStructGMRESSetup
c--------------------------------------------------------------------------
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


c--------------------------------------------------------------------------
c HYPRE_SStructGMRESSolve
c--------------------------------------------------------------------------
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


c--------------------------------------------------------------------------
c HYPRE_SStructGMRESSetKDim
c--------------------------------------------------------------------------
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


c--------------------------------------------------------------------------
c HYPRE_SStructGMRESSetTol
c--------------------------------------------------------------------------
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


c--------------------------------------------------------------------------
c HYPRE_SStructGMRESSetMinIter
c--------------------------------------------------------------------------
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


c--------------------------------------------------------------------------
c HYPRE_SStructGMRESSetMaxIter
c--------------------------------------------------------------------------
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


c--------------------------------------------------------------------------
c HYPRE_SStructGMRESSetStopCrit
c--------------------------------------------------------------------------
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


c--------------------------------------------------------------------------
c HYPRE_SStructGMRESSetPrecond
c--------------------------------------------------------------------------
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


c--------------------------------------------------------------------------
c HYPRE_SStructGMRESSetLogging
c--------------------------------------------------------------------------
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


c--------------------------------------------------------------------------
c HYPRE_SStructGMRESSetPrintLevel
c--------------------------------------------------------------------------
      subroutine fhypre_sstructgmressetprintlevel(fsolver, flevel)

      integer ierr
      integer flevel
      integer*8 fsolver

      call HYPRE_SStructGMRESSetPrintLevel(fsolver, flevel, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructgmressetprintlevel error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c HYPRE_SStructGMRESGetNumIterations
c--------------------------------------------------------------------------
      subroutine fhypre_sstructgmresgetnumiterati(fsolver, fnumiter)

      integer ierr
      integer fnumiter
      integer*8 fsolver

      call HYPRE_SStructGMRESGetNumIterations(fsolver, fnumiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructgmresgetnumiteration error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c HYPRE_SStructGMRESGetFinalRelativeResidualNorm
c--------------------------------------------------------------------------
      subroutine fhypre_sstructgmresgetfinalrelat(fsolver, fnorm)

      integer ierr
      integer*8 fsolver
      double precision fnorm

      call HYPRE_SStructGMRESGetFinalRelativeResidualNorm(fsolver, 
     1                                                    fnorm, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructgmresgetfinalrelative error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c HYPRE_SStructGMRESGetResidual
c--------------------------------------------------------------------------
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



c--------------------------------------------------------------------------
c HYPRE_SStructPCGCreate
c--------------------------------------------------------------------------
      subroutine fhypre_sstructpcgcreate(fcomm, fsolver)
{
      integer ierr
      integer*8 fcomm
      integer*8 fsolver

      call HYPRE_SStructPCGCreate(fcomm, fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructpcgcreate error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c HYPRE_SStructPCGDestroy
c--------------------------------------------------------------------------
      subroutine fhypre_sstructpcgdestroy(fsolver)

      integer ierr
      integer*8 fsolver

      call HYPRE_SStructPCGDestroy(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructpcgdestroy error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c HYPRE_SStructPCGSetup
c--------------------------------------------------------------------------
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


c--------------------------------------------------------------------------
c HYPRE_SStructPCGSolve
c--------------------------------------------------------------------------
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


c--------------------------------------------------------------------------
c HYPRE_SStructPCGSetTol
c--------------------------------------------------------------------------
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


c--------------------------------------------------------------------------
c HYPRE_SStructPCGSetMaxIter
c--------------------------------------------------------------------------
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


c--------------------------------------------------------------------------
c HYPRE_SStructPCGSetTwoNorm
c--------------------------------------------------------------------------
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


c--------------------------------------------------------------------------
c HYPRE_SStructPCGSetRelChange
c--------------------------------------------------------------------------
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


c--------------------------------------------------------------------------
c HYPRE_SStructPCGSetPrecond
c--------------------------------------------------------------------------
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


c--------------------------------------------------------------------------
c HYPRE_SStructPCGSetLogging
c--------------------------------------------------------------------------
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


c--------------------------------------------------------------------------
c HYPRE_SStructPCGSetPrintLevel
c--------------------------------------------------------------------------
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


c--------------------------------------------------------------------------
c HYPRE_SStructPCGGetNumIterations
c--------------------------------------------------------------------------
      subroutine fhypre_sstructpcggetnumiteration(fsolver, fnumiter)

      integer ierr
      integer fnumiter
      integer*8 fsolver

      call HYPRE_SStructPCGGetNumIterations(fsolver, fnumiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructpcggetnumiteration error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c HYPRE_SStructPCGGetFinalRelativeResidualNorm
c--------------------------------------------------------------------------
      subroutine fhypre_sstructpcggetfinalrelativ(fsolver, fnorm)

      integer ierr
      integer*8 fsolver
      double precision fnorm

      call HYPRE_SStructPCGGetFinalRelativeResidualNorm(fsolver, fnorm,
     1                                                  ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructpcggetfinalrelative error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c HYPRE_SStructPCGGetResidual
c--------------------------------------------------------------------------
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


c--------------------------------------------------------------------------
c HYPRE_SStructDiagScaleSetup
c--------------------------------------------------------------------------
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


c--------------------------------------------------------------------------
c HYPRE_SStructDiagScale
c--------------------------------------------------------------------------
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



c--------------------------------------------------------------------------
c HYPRE_SStructSplitCreate
c--------------------------------------------------------------------------
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


c--------------------------------------------------------------------------
c HYPRE_SStructSplitDestroy
c--------------------------------------------------------------------------
      subroutine fhypre_sstructsplitdestroy(fsolver)

      integer ierr
      integer*8 fsolver

      call HYPRE_SStructSplitDestroy(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsplitdestroy error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c HYPRE_SStructSplitSetup
c--------------------------------------------------------------------------
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


c--------------------------------------------------------------------------
c HYPRE_SStructSplitSolve
c--------------------------------------------------------------------------
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


c--------------------------------------------------------------------------
c HYPRE_SStructSplitSetTol
c--------------------------------------------------------------------------
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


c--------------------------------------------------------------------------
c HYPRE_SStructSplitSetMaxIter
c--------------------------------------------------------------------------
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


c--------------------------------------------------------------------------
c HYPRE_SStructSplitSetZeroGuess
c--------------------------------------------------------------------------
      subroutine fhypre_sstructsplitsetzeroguess(fsolver)

      integer ierr
      integer*8 fsolver

      call HYPRE_SStructSplitSetZeroGuess(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsplitsetzeroguess error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c HYPRE_SStructSplitSetNonZeroGuess
c--------------------------------------------------------------------------
      subroutine fhypre_sstructsplitsetnonzerogue(fsolver)

      integer ierr
      integer*8 fsolver

      call HYPRE_SStructSplitSetNonZeroGuess(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsplitsetnonzeroguess error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c HYPRE_SStructSplitSetStructSolver
c--------------------------------------------------------------------------
      subroutine fhypre_sstructsplitsetstructsolv(fsolver, fssolver)

      integer ierr
      integer fssolver
      integer*8 fsolver

      call HYPRE_SStructSplitSetStructSolver(fsolver, fssolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsplitsetstructsolve error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c HYPRE_SStructSplitGetNumIterations
c--------------------------------------------------------------------------
      subroutine fhypre_sstructsplitgetnumiterati(fsolver, fnumiter)

      integer ierr
      integer fnumiter
      integer*8 fsolver

      call HYPRE_SStructSplitGetNumIterations(fsolver, fnumiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsplitgetnumiteration error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c HYPRE_SStructSplitGetFinalRelativeResidualNorm
c--------------------------------------------------------------------------
      subroutine fhypre_sstructsplitgetfinalrelat(fsolver, fnorm)

      integer ierr
      integer*8 fsolver
      double precision fnorm

      call HYPRE_SStructSplitGetFinalRelativeResidualNorm(fsolver, 
     1                                                    fnorm, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsplitgetfinalrelative error = ', ierr
      endif

      return
      end



c--------------------------------------------------------------------------
c HYPRE_SStructSysPFMGCreate
c--------------------------------------------------------------------------
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


c--------------------------------------------------------------------------
c HYPRE_SStructSysPFMGDestroy
c--------------------------------------------------------------------------
      subroutine fhypre_sstructsyspfmgdestroy(fsolver)

      integer ierr
      integer*8 fsolver

      call HYPRE_SStructSysPFMGDestroy(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsyspfmgdestroy error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c HYPRE_SStructSysPFMGSetup
c--------------------------------------------------------------------------
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


c--------------------------------------------------------------------------
c HYPRE_SStructSysPFMGSolve
c--------------------------------------------------------------------------
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


c--------------------------------------------------------------------------
c HYPRE_SStructSysPFMGSetTol
c--------------------------------------------------------------------------
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


c--------------------------------------------------------------------------
c HYPRE_SStructSysPFMGSetMaxIter
c--------------------------------------------------------------------------
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


c--------------------------------------------------------------------------
c HYPRE_SStructSysPFMGSetRelChange
c--------------------------------------------------------------------------
      subroutine fhypre_sstructsyspfmgsetrelchang(fsolver, frel_change)

      integer ierr
      integer frel_change
      integer*8 fsolver

      call HYPRE_SStructSysPFMGSetRelChange(fsolver, frel_change, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsyspfmgsetrelchange error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c HYPRE_SStructSysPFMGSetZeroGuess
c--------------------------------------------------------------------------
      subroutine fhypre_sstructsyspfmgsetzerogues(fsolver)

      integer ierr
      integer*8 fsolver

      call HYPRE_SStructSysPFMGSetZeroGuess(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsyspfmgsetzeroguess error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c HYPRE_SStructSysPFMGSetNonZeroGuess
c--------------------------------------------------------------------------
      subroutine fhypre_sstructsyspfmgsetnonzerog(fsolver)

      integer ierr
      integer*8 fsolver

      call HYPRE_SStructSysPFMGSetNonZeroGuess(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsyspfmgsetnonzeroguess error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c HYPRE_SStructSysPFMGSetRelaxType
c--------------------------------------------------------------------------
      subroutine fhypre_sstructsyspfmgsetrelaxtyp(fsolver, frelax_type)

      integer ierr
      integer frelax_type
      integer*8 fsolver

      call HYPRE_SStructSysPFMGSetRelaxType(fsolver, frelax_type, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsyspfmgsetrelaxtype error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c HYPRE_SStructSysPFMGSetNumPreRelax
c--------------------------------------------------------------------------
      subroutine fhypre_sstructsyspfmgsetnumprere(fsolver, 
     1                                            fnum_pre_relax)

      integer ierr
      integer fnum_pre_relax
      integer*8 fsolver

      call HYPRE_SStructSysPFMGSetNumPreRelax(fsolver, fnum_pre_relax,
     1                                        ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsyspfmgsetnumprerelax error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c HYPRE_SStructSysPFMGSetNumPostRelax
c--------------------------------------------------------------------------
      subroutine fhypre_sstructsyspfmgsetnumpostr(fsolver, 
     1                                            fnum_post_relax)

      integer ierr
      integer fnum_post_relax
      integer*8 fsolver

      call HYPRE_SStructSysPFMGSetNumPostRelax(fsolver, fnum_post_relax,
     1                                        ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsyspfmgsetnumpostrelax error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c HYPRE_SStructSysPFMGSetSkipRelax
c--------------------------------------------------------------------------
      subroutine fhypre_sstructsyspfmgsetskiprela(fsolver, fskip_relax)

      integer ierr
      integer fskip_relax
      integer*8 fsolver

      call HYPRE_SStructSysPFMGSetSkipRelax(fsolver, fskip_relax, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsyspfmgsetskiprelax error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c HYPRE_SStructSysPFMGSetDxyz
c--------------------------------------------------------------------------
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


c--------------------------------------------------------------------------
c HYPRE_SStructSysPFMGSetLogging
c--------------------------------------------------------------------------
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


c--------------------------------------------------------------------------
c HYPRE_SStructSysPFMGSetPrintLevel
c--------------------------------------------------------------------------
      subroutine fhypre_sstructsyspfmgsetprintlev(fsolver, fprint_level)

      integer ierr
      integer fprint_level
      integer*8 fsolver

      call HYPRE_SStructSysPFMGSetPrintLevel(fsolver, fprint_level,
     1                                       ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsyspfmgsetprintlevel error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c HYPRE_SStructSysPFMGGetNumIterations
c--------------------------------------------------------------------------
      subroutine fhypre_sstructsyspfmggetnumitera(fsolver, fnumiter)

      integer ierr
      integer fnumiter
      integer*8 fsolver

      call HYPRE_SStructSysPFMGGetNumIterations(fsolver, fnumiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsyspfmggetnumiteration error = ', ierr
      endif

      return
      end


c--------------------------------------------------------------------------
c HYPRE_SStructSysPFMGGetFinalRelativeResidualNorm
c--------------------------------------------------------------------------
      subroutine fhypre_sstructsyspfmggetfinalrel(fsolver, fnorm)

      integer ierr
      integer*8 fsolver
      double precision fnorm

      call HYPRE_SStructSysPFMGGetFinalRelativeResidualNorm(fsolver, 
     1                                                      fnorm, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_sstructsyspfmggetfinalrelative error = ', ierr
      endif

      return
      end
