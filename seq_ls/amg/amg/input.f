c
      subroutine input(levmax,ecg,ncg,ewt,nwt,nstr,icdep,
     *                 ncyc,mu,ntrlx,iprlx,ierlx,iurlx,
     *                 ioutdat,ioutgrd,ioutmat,ioutres,ioutsol,
     *                 ipltgrd,ipltsol)
c
c---------------------------------------------------------------------
c
c     read amg setup and solve parameters
c
c---------------------------------------------------------------------
c
      implicit real*8 (a-h,o-z)
cveh      character*1 junk
c
c     setup parameters
c
      dimension icdep(10,10)
c
c=>   solution parameters
c
      dimension mu (25)
      dimension ntrlx(4)
      dimension iprlx(4)
      dimension ierlx(4)
      dimension iurlx(4)
c
c---------------------------------------------------------------------
c
      open(5,file='amg.dat',status='old')
c
c     read setup data
c
      read(5,*) junk

      read(5,*) levmax
      read(5,*) ncg,ecg
      read(5,*) nwt,ewt
      read(5,*) nstr
c
c     read solve data
c
      read(5,*) junk
c
      read(5,*) ncyc
      read(5,*) (mu(k),k=1,10)
      read(5,*) (ntrlx(i),i=1,4)
      read(5,*) (iprlx(i),i=1,4)
      read(5,*) (ierlx(i),i=1,4)
      read(5,*) (iurlx(i),i=1,4)
c
c     read output data 
c
      read(5,*) junk

      read(5,*) ioutdat
      read(5,*) ioutgrd
      read(5,*) ioutmat
      read(5,*) ioutres
      read(5,*) ioutsol
c
c     read(5,*) ipltgrd
c     read(5,*) ipltsol
c
      close(5)
      return
      end
