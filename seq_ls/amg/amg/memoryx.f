c
      subroutine memoryx(k,imin,imax,ia,ib,ipmn,ipmx)
c
c---------------------------------------------------------------------
c
c     set up amg components
c
c---------------------------------------------------------------------
c
      implicit real*8 (a-h,o-z)
c
      dimension imin(25),imax(25)
      dimension ia (*)

      dimension ipmn(25),ipmx(25)

      dimension ib (*)
c
c---------------------------------------------------------------------
c
      write(6,999)
  999 format(/'  level ',i2,' storage requirements'/)
c
c     point storage
c
      write(6,1000)
 1000 format(/'  point information '/)
      iplo=ipmn(k)
      iphi=ipmx(k)
      write(6,1002) iplo,iphi
 1002 format('  xp     - ',2i10)
      write(6,1003) iplo,iphi
 1003 format('  yp     - ',2i10)
      write(6,1005) iplo,iphi+1
 1005 format('  iv     - ',2i10)
c
c     variable storage
c
      write(6,2000)
 2000 format(/'  variable information '/)
      ilo=imin(k)
      ihi=imax(k)
      write(6,2001) ilo,ihi
 2001 format('  u      - ',2i10)
      write(6,2002) ilo,ihi
 2002 format('  f      - ',2i10)
      write(6,2005) ilo,ihi
 2005 format('  icg    - ',2i10)
      write(6,2006) ilo,ihi
 2006 format('  ifg    - ',2i10)
      write(6,2007) ilo,ihi
 2007 format('  ip     - ',2i10)
      write(6,2006) ilo,ihi
 2009 format('  icgfix - ',2i10)
c
c     matrix storage
c
      write(6,3000)
 3000 format(/'  matrix information '/)
      write(6,2003) ilo,ihi+1
 2003 format('  ia     - ',2i10)
      jlo=ia(imin(k))
      jhi=ia(imax(k)+1)-1
      write(6,3001) jlo,jhi
 3001 format('  a      - ',2i10)
      write(6,3002) jlo,jhi+1
 3002 format('  ja     - ',2i10)
c
c     interpolation storage
c
      write(6,4000)
 4000 format(/'  interpolation information '/)
      write(6,2004) ilo,ihi+1
 2004 format('  ib     - ',2i10)
      jlo=ib(imin(k))
      jhi=ib(imax(k)+1)-1
      write(6,4001) jlo,jhi
 4001 format('  b      - ',2i10)
      write(6,4002) jlo,jhi+1
 4002 format('  jb     - ',2i10)
      return
      end
