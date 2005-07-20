c--------------------------------------------------------------------------
c GenerateLaplacian
c--------------------------------------------------------------------------
      subroutine fhypre_generatelaplacian(fcom, fnx, fny, fnz,
     1                                    fcapp, fcapq, fcapr, 
     1                                    fp, fq, fr,
     2                                    fvalue, fmatrix)

      integer ierr
      integer fcomm
      integer fnx, fny, fnz
      integer fcapp, fcapq, fcapr
      integer fp, fq, fr
      double precision fvalue
      integer*8 fmatrix

      call HYPRE_GenerateLaplacian(fcomm, fnx, fny, fnz, fcapp, fcapq,
     1                       fcapr, fp, fq, fr, fvalue, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_generatelaplacian error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGCreate
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgcreate(fsolver)

      integer ierr
      integer*8 fsolver

      call HYPRE_BoomerAMGCreate(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgcreate error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGDestroy
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgdestroy(fsolver)

      integer ierr
      integer*8 fsolver

      call HYPRE_BoomerAMGDestroy(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgdestroy error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetup
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetup(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_BoomerAMGSetup(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetup error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSolve
c-------------------------------------------------------------------------- 
      subroutine fhypre_boomeramgsolve(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_BoomerAMGSolve(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsolve error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSolveT
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsolvet(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_BoomerAMGSolveT(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsolvet error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetRestriction
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetrestriction(fsolver, frestr_par)

      integer ierr
      integer frestr_par
      integer*8 fsolver

      call HYPRE_BoomerAMGSetRestriction(fsolver, frestr_par, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetrestriction error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetMaxLevels
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetmaxlevels(fsolver, fmaxlvl)

      integer ierr
      integer fmaxlvl
      integer*8 fsolver

      call HYPRE_BoomerAMGSetMaxLevels(fsolver, fmaxlvl, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetmaxlevels error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetStrongThreshold
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetstrongthrshl(fsolver, fstrong)

      integer ierr
      double precision fstrong
      integer*8 fsolver

      call HYPRE_BoomerAMGSetStrongThrshld(fsolver, fstrong, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetstrongthreshold error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetMaxRowSum
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetmaxrowsum(fsolver, fmaxrowsum)

      integer ierr
      double precision fmaxrowsum
      integer*8 fsolver

      call HYPRE_BoomerAMGSetMaxRowSum(fsolver, fmaxrowsum, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetmaxrowsum error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetTruncFactor
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsettruncfactor(fsolver, ftrunc_factor)

      integer ierr
      double precision ftrunc_factor
      integer*8 fsolver

      call HYPRE_BoomerAMGSetTruncFactor(fsolver, ftrunc_factor, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsettruncfactor error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetInterpType
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetinterptype(fsolver finterp)

      integer ierr
      integer finterp
      integer*8 fsolver

      call HYPRE_BoomerAMGSetInterpType(fsolver, finterp, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetinterptype error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetMinIter
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetminiter(fsolver, fminiter)

      integer ierr
      integer fminiter  
      integer*8 fsolver

      call HYPRE_BoomerAMGSetMinIter(fsolver, fminiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetminiter error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetMaxIter
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetmaxiter(fsolver, fmaxiter)

      integer ierr
      integer fmaxiter  
      integer*8 fsolver

      call HYPRE_BoomerAMGSetMaxIter(fsolver, fmaxiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetmaxiter error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetCoarsenType
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetcoarsentype(fsolver, fcoarsen)

      integer ierr
      integer fcoarsen
      integer*8 fsolver

      call HYPRE_BoomerAMGSetCoarsenType(fsolver, fcoarsen, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetcoarsentype error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetMeasureType
c--------------------------------------------------------------------------

      subroutine fhypre_boomeramgsetmeasuretype(fsolver, fmeasure)

      integer ierr
      integer fmeasure
      integer*8 fsolver

      call HYPRE_BoomerAMGSetMeasureType(fsolver, fmeasure, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetmeasuretype error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetSetupType
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetsetuptype(fsolver, fsetup)

      integer ierr
      integer fsetup
      integer*8 fsolver

      call HYPRE_BoomerAMGSetSetupType(fsolver, fsetup, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetuptype error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetCycleType
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetcycletype(fsolver, fcycle)

      integer ierr
      integer fcycle
      integer*8 fsolver

      call HYPRE_BoomerAMGSetCycleType(fsolver, fcycle, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetcycletype error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetTol
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsettol(fsolver, ftol)

      integer ierr
      double precision ftol
      integer*8 fsolver

      call HYPRE_BoomerAMGSetTol(fsolver, ftol, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsettol error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetNumSweeps
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetnumsweeps(fsolver, fnumsweeps)

      integer ierr
      integer fnumsweeps
      integer*8 fsolver

      call HYPRE_BoomerAMGSetNumSweeps(fsolver, fnumsweeps, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetnumsweeps error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetCycleNumSweeps
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetcyclenumswee(fsolver, fnumsweeps,
     1                                             fk)

      integer ierr
      integer fnumsweeps
      integer fk
      integer*8 fsolver

      call HYPRE_BoomerAMGSetCycleNumSweeps(fsolver, fnumsweeps, fk,
     1                                      ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetcyclenumsweeps error: ', ierr
      endif

      return
      end
c--------------------------------------------------------------------------
c HYPRE_BoomerAMGInitGridRelaxation
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramginitgridrelaxat(fnumsweeps, fgridtype,
     1                                            fgridrelax, fcoarsen,
     2                                            frelaxwt, fmaxlvl)

      integer ierr
      integer fcoarsen
      integer fmaxlvl
      integer*8 fnumsweeps
      integer*8 fgridtype
      integer*8 fgridrelax
      integer*8 frelaxwt

      call HYPRE_BoomerAMGInitGridRelaxatn(fnumsweeps, fgridtype, 
     1                                       fgridrelax, fcoarsen, 
     2                                       frelaxwt, fmaxlvl, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramginitgridrelaxation error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGFinalizeGridRelaxation
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgfingridrelaxatn(fnumsweeps, fgridtype,
     1                                            fgridrelax, frelaxwt)

      integer ierr
      integer*8 fnumsweeps
      integer*8 fgridtype
      integer*8 fgridrelax
      integer*8 frelaxwt

c     hypre_TFree(num_grid_sweeps);
c     hypre_TFree(grid_relax_type);
c     hypre_TFree(grid_relax_points);
c     hypre_TFree(relax_weights);

      ierr = 0

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetRelaxType
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetrelaxtype(fsolver, frelaxtype)

      integer ierr
      integer frelaxtype
      integer*8 fsolver

      call HYPRE_BoomerAMGSetRelaxType(fsolver, frelaxtype, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetrelaxtype error: ', ierr
      endif
 
      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetCycleRelaxType
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetcyclerelaxty(fsolver, frelaxtype,
     1                                             fk)

      integer ierr
      integer frelaxtype
      integer fk
      integer*8 fsolver

      call HYPRE_BoomerAMGSetCycleRelaxType(fsolver, fk, frelaxtype, 
     1                                      ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetcyclerelaxtype error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetRelaxWt
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetrelaxwt(fsolver, frelaxwt)
     
      integer ierr
      integer*8 fsolver
      double precision frelaxwt

      call HYPRE_BoomerAMGSetRelaxWt(fsolver, frelaxwt, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetrelaxwt error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetLevelRelaxWt
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetlevelrelaxwt(fsolver, frelaxwt, 
     1                                           flevel)

      integer ierr
      integer flevel
      integer*8 fsolver
      double precision frelaxwt

      call HYPRE_BoomerAMGSetLevelRelaxWt(fsolver, frelaxwt, flevel,
     1                                    ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetlevelrelaxwt error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetPrintLevel
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetprintlevel(fsolver, fprintlevel)

      integer ierr
      integer fprintlevel
      integer*8 fsolver

      call HYPRE_BoomerAMGSetPrintLevel(fsolver, fprintlevel, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetprintlevel error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetPrintFileName
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetprintfilenam(fsolver, fname)

      integer ierr
      integer*8 fsolver
      character*(*) fname

      call HYPRE_BoomerAMGSetPrintFileName(fsolver, fname, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetprintfilename error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetDebugFlag
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetdebugflag(fsolver, fdebug)

      integer ierr
      integer fdebug
      integer*8 fsolver

      call HYPRE_BoomerAMGSetDebugFlag(fsolver, fdebug, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetdebugflag error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGGetNumIterations
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramggetnumiterations(fsolver, fnumiter)

      integer ierr
      integer fnumiter
      integer*8 fsolver

      call HYPRE_BoomerAMGGetNumIterations(fsolver, fnumiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramggetnumiterations error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGGetFinalRelativeRes
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramggetfinalreltvre(fsolver, frelresid)

      integer ierr
      integer*8 fsolver
      double precision frelresid

      call HYPRE_BoomerAMGGetFinalReltvRes(fsolver, frelresid, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramggetfinalrelativeres error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_BoomerAMGSetGSMG
c--------------------------------------------------------------------------
      subroutine fhypre_boomeramgsetgsmg(fsolver, fgsmg)

      integer ierr
      integer fgsmg
      integer*8 fsolver

      call HYPRE_BoomerAMGSetGSMG(fsolver, fgsmg, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_boomeramgsetgsmg error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_ParCSRBiCGSTABCreate
c--------------------------------------------------------------------------
      subroutine fhypre_parcsrbicgstabcreate(fcomm, fsolver)

      integer fcomm
      integer ierr
      integer*8 fsolver

      call HYPRE_ParCSRBiCGSTABCreate(fcomm, fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrbicgstabcreate error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_ParCSRBiCGSTABDestroy
c-------------------------------------------------------------------------- 
      subroutine fhypre_parcsrbicgstabdestroy(fsolver)

      integer ierr
      integer*8 fsolver

      call HYPRE_ParCSRBiCGSTABDestroy(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrbicgstabdestroy error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_ParCSRBiCGSTABSetup
c------------------------------------------------------------------------- 
      subroutine fhypre_parcsrbicgstabsetup(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_ParCSRBiCGSTABSetup(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrbicgstabsetup error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_ParCSRBiCGSTABSolve
c-------------------------------------------------------------------------- 
      subroutine fhypre_parcsrbicgstabsolve(fsolver, fA, fb, fx)
     
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_ParCSRBiCGSTABSolve(fsolver, fA, fb, fx)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrbicgstabsolve error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRBiCGSTABSetTol
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrbicgstabsettol(fsolver, ftol)

      integer ierr
      integer*8 fsolver
      double precision ftol

      call HYPRE_ParCSRBiCGSTABSetTol(fsolver, ftol, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrbicgstabsettol error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRBiCGSTABSetMinIter
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrbicgstabsetminiter(fsolver, fminiter)

      integer ierr
      integer fminiter
      integer*8 fsolver

      call HYPRE_ParCSRBiCGSTABSetMinIter(fsolver, fminiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrparcsrbicgstabsetminiter error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRBiCGSTABSetMaxIter
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrbicgstabsetmaxiter(fsolver, fmaxiter)

      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call HYPRE_ParCSRBiCGSTABSetMaxIter(fsolver, fmaxiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrbicgstabsetmaxiter error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRBiCGSTABSetPrecond
c--------------------------------------------------------------------------
      subroutine fhypre_parcsrbicgstabsetprecond(fsolver, fprecond_id, 
     1                                           fprecond)       

      integer ierr
      integer fprecond_id
      integer*8 fsolver
      integer*8 fprecond

      call HYPRE_ParCSRBiCGSTABSetPrecond(fsolver, fprecond_id, 
     1                                    fprecond, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrbicgstabsetprecond error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRBiCGSTABGetPrecond
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrbicgstabgetprecond(fsolver, fprecond)
      
      integer ierr
      integer*8 fsolver
      integer*8 fprecond

      call HYPRE_ParCSRBiCGSTABGetPrecond(fsolver, fprecond)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrbicgstabgetprecond error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRBiCGSTABSetLogging
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrbicgstabsetlogging(fsolver, flog)

      integer ierr
      integer flog
      integer*8 fsolver

      call HYPRE_ParCSRBiCGSTABSetLogging(fsolver, flog, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrbicgstabsetlogging error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRBiCGSTABGetNumIter
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrbicgstabgetnumiter(fsolver, fnumiter)

      integer ierr
      integer fnumiter
      integer*8 fsolver

      call HYPRE_ParCSRBiCGSTABGetNumIter(fsolver, fnumiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrbicgstabgetnumiterations error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRBiCGSTABGetFinalRel
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrbicgstabgetfinalre(fsolver, fnorm)

      integer ierr
      integer*8 fsolver
      double precision fnorm

      call HYPRE_ParCSRBiCGSTABGetFinalRel(fsolver, fnorm, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrbicgstabgetfinalrel error: ', ierr
      endif
      return
      end



c-------------------------------------------------------------------------
c HYPRE_ParCSRCGNRCreate
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrcgnrcreate(fcomm, fsolver)

      integer ierr
      integer fcomm
      integer*8 fsolver

      call HYPRE_ParCSRCGNRCreate(fcomm, fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrcgnrcreate error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRCGNRDestroy
c------------------------------------------------------------------------- 
      subroutine fhypre_parcsrcgnrdestroy(fsolver)

      integer ierr
      integer*8 fsolver

      call HYPRE_ParCSRCGNRDestroy(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrcgnrdestroy error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRCGNRSetup
c------------------------------------------------------------------------- 
      subroutine fhypre_parcsrcgnrsetup(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_ParCSRCGNRSetup(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrcgnrsetup error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRCGNRSolve
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrcgnrsolve(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_ParCSRCGNRSolve(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrcgnrsolve error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRCGNRSetTol
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrcgnrsettol(fsolver, ftol)

      integer ierr
      integer*8 fsolver
      double precision ftol

      call HYPRE_ParCSRCGNRSetTol(fsolver, ftol, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrcgnrsettol error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRCGNRSetMaxIter
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrcgnrsetmaxiter(fsolver, fmaxiter)
 
      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call HYPRE_ParCSRCGNRSetMaxIter(fsolver, fmaxiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrcgnrsetmaxiter error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRCGNRSetPrecond
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrcgnrsetprecond(fsolver, fprecond_id, 
     1                                       fprecond)

      integer ierr
      integer fprecond_id
      integer*8 fsolver
      integer*8 fprecond

      call HYPRE_ParCSRCGNRSetPrecond(fsolver, fprecond_id, fprecond, 
     1                                ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrcgnrsetprecond error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRCGNRGetPrecond
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrcgnrgetprecond(fsolver, fprecond)

      integer ierr
      integer*8 fsolver
      integer*8 fprecond

      call HYPRE_ParCSRCGNRGetPrecond(fsolver, fprecond, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrcgnrgetprecond error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRCGNRSetLogging
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrcgnrsetlogging(fsolver, flog)

      integer ierr
      integer flog
      integer*8 fsolver

      call HYPRE_ParCSRCGNRSetLogging(fsolver, flog, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrcgnrsetlogging error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRCGNRGetNumIteration
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrcgnrgetnumiteratio(fsolver, fnumiter)

      integer ierr
      integer fnumiter
      integer*8 fsolver

      call HYPRE_ParCSRCGNRGetNumIteration(fsolver, fnumiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrcgnrgetnumiterations error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRCGNRGetFinalRelativ
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrcgnrgetfinalrelati(fsolver, fnorm)

      integer ierr
      integer*8 fsolver
      double precision fnorm

      call HYPRE_ParCSRCGNRGetFinalRelativ(fsolver, fnorm, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrcgnrgetfinalrelativ error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRGMRESCreate
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrgmrescreate(fcomm, fsolver)

      integer ierr
      integer fcomm
      integer*8 fsolver

      call HYPRE_ParCSRGMRESCreate(fcomm, fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrgmrescreate error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRGMRESDestroy
c------------------------------------------------------------------------- 
      subroutine fhypre_parcsrgmresdestroy(fsolver)

      integer ierr
      integer*8 fsolver

      call HYPRE_ParCSRGMRESDestroy(fsolver)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrgmresdestroy error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRGMRESSetup
c------------------------------------------------------------------------- 
      subroutine fhypre_parcsrgmressetup(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_ParCSRGMRESSetup(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrgmressetup error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRGMRESSolve
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrgmressolve(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_ParCSRGMRESSolve(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrgmressolve error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRGMRESSetKDim
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrgmressetkdim(fsolver, fkdim)

      integer ierr
      integer fkdim
      integer*8 fsolver

      call HYPRE_ParCSRGMRESSetKDim(fsolver, fkdim, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrgmressetkdim error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRGMRESSetTol
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrgmressettol(fsolver, ftol)

      integer ierr
      integer*8 fsolver
      double precision ftol

      call HYPRE_ParCSRGMRESSetTol(fsolver, ftol, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrgmressettol error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRGMRESSetMinIter
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrgmressetminiter(fsolver, fminiter)

      integer ierr
      integer fminiter
      integer*8 fsolver

      call HYPRE_ParCSRGMRESSetMinIter(fsolver, fminiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrgmressetminiter error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRGMRESSetMaxIter
c--------------------------------------------------------------------------
      subroutine fhypre_parcsrgmressetmaxiter(fsolver, fmaxiter)

      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call HYPRE_ParCSRGMRESSetMaxIter(fsolver, fmaxiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrgmressetmaxiter error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRGMRESSetPrecond
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrgmressetprecond(fsolver, fprecond_id, 
     1                                        fprecond)

      integer ierr
      integer fprecond_id
      integer*8 fsolver
      integer*8 fprecond

      call HYPRE_ParCSRGMRESSetPrecond(fsolver, fprecond_id, fprecond,
     1                                 ierr)
     
      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrgmressetprecond error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRGMRESGetPrecond
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrgmresgetprecond(fsolver, fprecond)

      integer ierr
      integer*8 fsolver
      integer*8 fprecond

      call HYPRE_ParCSRGMRESGetPrecond(fsolver, fprecond, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrgmresgetprecond error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRGMRESSetLogging
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrgmressetlogging(fsolver, flog)

      integer ierr
      integer flog
      integer*8 fsolver

      call HYPRE_ParCSRGMRESSetLogging(fsolver, flog, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrgmressetlogging error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRGMRESGetNumIter
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrgmresgetnumiterati(fsolver, fnumiter)

      integer ierr
      integer fnumiter
      integer*8 fsolver

      call HYPRE_ParCSRGMRESGetNumIteratio(fsolver, fnumiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrgmresgetnumiterations error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRGMRESGetFinalRelati
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrgmresgetfinalrelat(fsolver, fnorm)

      integer ierr
      integer*8 fsolver
      double precision fnorm

      call HYPRE_ParCSRGMRESGetFinalRelati(fsolver, fnorm, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrgmresgetfinalrelative error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParaSailsCreate
c-------------------------------------------------------------------------
      subroutine fhypre_parasailscreate(fcomm, fsolver)

      integer ierr
      integer fcomm
      integer*8 fsolver

      call HYPRE_ParaSailsCreate(fcomm, fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parasailscreate error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParaSailsDestroy
c-------------------------------------------------------------------------
      subroutine fhypre_parasailsdestroy(fsolver)

      integer ierr
      integer*8 fsolver

      call HYPRE_ParaSailsDestroy(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parasailsdestroy error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParaSailsSetup
c------------------------------------------------------------------------- 
      subroutine fhypre_parasailssetup(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_ParaSailsSetup(fsolver, fA, fb, fx, ierr) 

      if(ierr .ne. 0) then
         print *, 'fhypre_parasailssetup error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParaSailsSolve
c------------------------------------------------------------------------- 
      subroutine fhypre_parasailssolve(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_ParaSailsSolve(fsolver, fA, fb, fx, ierr) 

      if(ierr .ne. 0) then
         print *, 'fhypre_parasailssolve error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParaSailsSetParams
c-------------------------------------------------------------------------
      subroutine fhypre_parasailssetparams(fsolver, fthresh, fnlevels)

      integer ierr
      integer fnlevels
      integer*8 fsolver
      double precision fthresh

      call HYPRE_ParaSailsSetParams(fsolver, fthresh, fnlevels, ierr) 

      if(ierr .ne. 0) then
         print *, 'fhypre_parasailssetparams error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParaSailsSetFilter
c-------------------------------------------------------------------------
      subroutine fhypre_parasailssetfilter(fsolver, ffilter)

      integer ierr
      integer*8 fsolver
      double precision ffilter

      call HYPRE_ParaSailsSetFilter(fsolver, ffilter, ierr) 

      if(ierr .ne. 0) then
         print *, 'fhypre_parasailssetfilter error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParaSailsSetSym
c-------------------------------------------------------------------------
      subroutine fhypre_parasailssetsym(fsolver, fsym)

      integer ierr
      integer fsym
      integer*8 fsolver

      call HYPRE_ParaSailsSetSym(fsolver, fsym, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parasailssetsym error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParaSailsSetLogging
c-------------------------------------------------------------------------
      subroutine fhypre_parasailssetlogging(fsolver, flog)

      integer ierr
      integer flog
      integer*8 fsolver

      call HYPRE_ParaSailsSetLogging(fsolver, flog, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parasailssetlogging error: ', ierr
      endif
      
      return
      end



c-------------------------------------------------------------------------
c HYPRE_ParCSRPCGCreate
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrpcgcreate(fcomm, fsolver)

      integer ierr
      integer fcomm
      integer*8 fsolver

      call HYPRE_ParCSRPCGCreate(fcomm, fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrpcgcreate error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRPCGDestroy
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrpcgdestroy(fsolver)

      integer ierr
      integer*8 fsolver

      call HYPRE_ParCSRPCGDestroy(fsolver)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrpcgdestroy error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRPCGSetup
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrpcgsetup(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_ParCSRPCGSetup(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrpcgcreate error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRPCGSolve
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrpcgsolve(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_ParCSRPCGSolve(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrpcgsolve error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRPCGSetTol
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrpcgsettol(fsolver, ftol)

      integer ierr
      integer*8 fsolver
      double precision ftol

      call HYPRE_ParCSRPCGSetTol(fsolver, ftol, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrpcgsettol error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRPCGSetMaxIter
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrpcgsetmaxiter(fsolver, fmaxiter)

      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call HYPRE_ParCSRPCGSetMaxIter(fsolver, fmaxiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrpcgsetmaxiter error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRPCGSetTwoNorm
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrpcgsettwonorm(fsolver, ftwonorm)

      integer ierr
      integer ftwonorm
      integer*8 fsolver

      call HYPRE_ParCSRPCGSetTwoNorm(fsolver, ftwonorm, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrpcgsettwonorm error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRPCGSetRelChange
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrpcgsetrelchange(fsolver, frelchange)

      integer ierr
      integer frelchange
      integer*8 fsolver

      call HYPRE_ParCSRPCGSetRelChange(fsolver, frelchange, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrpcgsetrelchange error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRPCGSetPrecond
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrpcgsetprecond(fsolver, fprecond_id, 
     1                                      fprecond)

      integer ierr
      integer fprecond_id
      integer*8 fsolver
      integer*8 fprecond

      call HYPRE_ParCSRPCGSetPrecond(fsolver, fprecond_id, fprecond, 
     1                              ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrpcgsetprecond error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRPCGGetPrecond
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrpcggetprecond(fsolver, fprecond)

      integer ierr
      integer*8 fsolver
      integer*8 fprecond

      call HYPRE_ParCSRPCGGetPrecond(fsolver, fprecond, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrpcggetprecond error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRPCGSetPrintLevel
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrpcgsetprintlevel(fsolver, flevel)

      integer ierr
      integer flevel
      integer*8 fsolver

      call HYPRE_ParCSRPCGSetPrintLevel(fsolver, flevel, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrpcgsetprintlevel error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRPCGGetNumIterations
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrpcggetnumiteration(fsolver, fnumiter)

      integer ierr
      integer fnumiter
      integer*8 fsolver

      call HYPRE_ParCSRPCGGetNumIterations(fsolver, fnumiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrpcggetnumiteration error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRPCGGetFinalRelativeResidualNorm
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrpcggetfinalrelativ(fsolver, fnorm)

      integer ierr
      integer*8 fsolver
      double precision fnorm

      call HYPRE_ParCSRPCGGetFinalRelative(fsolver, fnorm, ierr)
      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrpcggetfinalrelative error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRDiagScaleSetup
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrdiagscalesetup(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_ParCSRDiagScaleSetup(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrdiagscalesetup error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRDiagScale
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrdiagscale(fsolver, fHA, fHy, fHx)

      integer ierr
      integer*8 fsolver
      integer*8 fHA
      integer*8 fHy
      integer*8 fHx

      call HYPRE_ParCSRDiagScale(fsolver, fHA, fHy, fHx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrdiagscale error: ', ierr
      endif

      return
      end



c-------------------------------------------------------------------------
c HYPRE_ParCSRPilutCreate
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrpilutcreate(fcomm, fsolver)

      integer ierr
      integer fcomm
      integer*8 fsolver

      call HYPRE_ParCSRPilutCreate(fcomm, fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrpilutcreate error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRPilutDestroy
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrpilutdestroy(fsolver)

      integer ierr
      integer*8 fsolver

      call HYPRE_ParCSRPilutDestroy(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrpilutdestroy error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRPilutSetup
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrpilutsetup(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_ParCSRPilutSetup(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrpilutsetup error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRPilutSolve
c------------------------------------------------------------------------- 
      subroutine fhypre_parcsrpilutsolve(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call HYPRE_ParCSRPilutSolve(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrpilutsolve error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRPilutSetMaxIter
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrpilutsetmaxiter(fsolver, fmaxiter)

      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call HYPRE_ParCSRPilutSetMaxIter(fsolver, fmaxiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrpilutsetmaxiter error: ', ierr
      endif

      return
      end

c-------------------------------------------------------------------------
c HYPRE_ParCSRPilutSetDropToleran
c-------------------------------------------------------------------------
      subroutine fhypre_parcsrpilutsetdroptolera(fsolver, ftol)

      integer ierr
      integer*8 fsolver
      double precision ftol

      call HYPRE_ParCSRPilutSetDropToleran(fsolver, ftol, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrpilutsetdroptol error: ', ierr
      endif

      return
      end

c--------------------------------------------------------------------------
c HYPRE_ParCSRPilutSetFacRowSize
c--------------------------------------------------------------------------
      subroutine fhypre_parcsrpilutsetfacrowsize(fsolver, fsize)
      
      integer ierr
      integer fsize
      integer*8 fsolver

      call HYPRE_ParCSRPilutSetFacRowSize(fsolver, fsize, ierr)

      if(ierr .ne. 0) then
         print *, 'fhypre_parcsrpilutsetfacrowsize error: ', ierr
      endif

      return
      end

