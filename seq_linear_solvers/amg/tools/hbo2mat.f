      program hbo2mat
c
c     This Fortran program reads a file of sparse matrices in
c     Harwell-Boeing format and creates several MATLAB .mat files
c     containing the individual sparse matrices and associated data.
c
c     The curators of the Harwell-Boeing Sparse Matrix collection 
c     are Iain Duff, Roger Grimes and John Lewis.  Duff started the
c     project when he was at Harwell Laboratory in Britain, but he
c     has since moved to the Rutherford Appleton Laboratory.  Grimes
c     and Lewis are with Boeing Computer Services.
c
c     The collection is available via anonymous FTP from
c         orion.cerfacs.fr (130.63.200.33)
c     in directory
c         pub/harwell_boeing
c     A compressed, postscript file contained a detailed guide is
c         pub/harwell_boeing/userguide.ps.Z
c     Binary mode must be used for most of the file transfers.
c     
c     Once a data file has been obtained from CERFACS, this program
c     converts it to a form accessible by MATLAB.  Any command line
c     argument is taken as the input file name.  (Standard input is used
c     if there is no command line argument.)  For example, the command
c
c         hbo2mat smtape.data
c
c     processes one of the files, which happens to contain 36 matrices.
c     The result is the 36 .mat files, with names like gent113.mat and
c     abb313.mat, that make up the original Harwell collection.
c     Each file contains one sparse matrix, as well as auxilliary data
c     including matrix name, the title, the type and any right hand
c     sides and exact solutions.
c
c     This program uses the MATLAB External Interface Library.  
c     A Unix Makefile would look something like:
c
c        LIBMAT = .../matlab/extern/lib/<arch>/libmat.a
c        hbo2mat: hbo2mat.f
c        f77 hbo2mat.f $(LIBMAT) -o hbo2mat 
c
c     The MATLAB function "hbo" can be used to load the resulting .mat
c     files, one at a time, and, where appropriate, create the symmetric
c     matrix from its lower triangle.
c
c     Cleve Moler, The MathWorks, 4/2/94.
c
      parameter (NMAX = 15500, NZMAX = 550000)
c     These values are large enough for all matrices in the collection
c     as of 3/28/94, except bcsstk30, 31 and 32 in bcsstuc5.data.
c
      real*8 a(NZMAX)
      real*8 hbfill, hbzero, amax
      integer ia(NZMAX),ja(NMAX)
      integer fp,ap,bp,xp,gp
      logical aval, acmplx, atype
      integer totcrd, ptrcrd, indcrd, valcrd, rhscrd, m, n,
     >   nnz, neltvl, nmax, nzmax, nrhs, nnzrhs, mnrhs
      character title*72, key*8, type*3, ptrfmt*16, indfmt*16,
     >   valfmt*20, rhsfmt*20, rhstype*3, name*8, filename*12, c*1,
     >   infilename*32
c
      if (iargc() .gt. 0) then
         call getarg(1,infilename)
         iounit = 8
         open(unit=iounit,file=infilename)
      else
         iounit = 5
      endif
c
c     Start loop through file.
c     Loop is terminated when end of file is detected.
c
      do while (.true.)
c
         read (iounit, 10, end=99) title, key,
     >         totcrd, ptrcrd, indcrd, valcrd, rhscrd,
     >         type, m, n, nnz, neltvl,
     >         ptrfmt, indfmt, valfmt, rhsfmt
 10      format (a72, a8 / 5i14 / a3, 11x, 4i14 / 2a16, 2a20)
         if (rhscrd .eq. 0) then 
            rhstype = 'NUN'
            nrhs = 0
            nnzrhs = 0
         else
            read (iounit, 20) rhstype, nrhs, nnzrhs
 20         format (a3,11x,2i14)
         endif
c
c        Matrix name is nonblank characters in columns 73:80
c        of the title line, converted to lower case.
c
         k = 0
         name = '        '
         do j = 1, 8
            c = key(j:j)
            if (c .ne. ' ') then
               k = k + 1
               if (c.ge.'A' .and. c.le.'Z') c = char(int(c)+32)
               name(k:k) = c
            endif
         enddo
         filename = name
         filename(k+1:k+4) = '.mat'
         print '(a12,3x,a8,3x,a3,3x,a3,3x,5i8)',
     >      filename, key, type, rhstype, m, n, nnz, nrhs, neltvl
c
         if (n.gt.nmax-1 .or. nnz.gt.nzmax) then
            print *, 'Matrix parameters exceed allocated storage.'
            print *, 'm, n, nnz = ', m, n, nnz
            print *, 'nmax, nzmax = ', nmax, nzmax
            stop
         endif
c
c        Open file and insert descriptive data
c
         fp = matOpen(filename,'w')
         if (fp .eq. 0) print *, 'Could not open file', filename
         isok = matPutString(fp, 'hbtitle', title)
         isok = matPutString(fp, 'hbname', name)
         isok = matPutString(fp, 'hbtype', type)
c
c        Read the matrix
c
         read (iounit,ptrfmt) (ja (i), i = 1, n+1)
         read (iounit,indfmt) (ia (i), i = 1, nnz)
         aval = valcrd .gt. 0
         acmplx = type(1:1) .eq. 'C'
         atype = type(3:3) .eq. 'A'
         if (aval) then
            if (acmplx) then
               read (iounit,valfmt) (a(i), a(nnz+i), i = 1, nnz)
            else if (atype) then
               read (iounit,valfmt) (a(i), i = 1, nnz)
            else
               read (iounit,valfmt) (a(i), i = 1, neltvl)
            endif
         endif
c
c        Store the matrix in the file
c
         if (atype) then
c
c           Assembled, sparse matrix.
c
c           Fill in fixed value if none specified.
c
            if (.not. aval) then
               hbfill = 255
               do k = 1, nnz
                  a(k) = hbfill
               enddo
               isok = matPutFull(fp,'hbfill',1,1,hbfill,0.0d0)
            endif
c
c           Replace explicit zeros by value large enough to be distinct,
c           but small enough to preserve any compact storage.
c  
            if (aval) then
               hbzero = 0
               amax = a(1)
               do k = 1, nnz
                  if (a(k) .eq. 0.0d0) hbzero = 1.0d0
                  amax = dmax1(amax,a(k))
               enddo
               if (hbzero .ne. 0) then
                  hbzero = 2*amax
                  if (amax .lt. 65535) hbzero = 65535
                  if (amax .lt. 255) hbzero = 255
                  do k = 1, nnz
                     if (a(k) .eq. 0.0d0) a(k) = hbzero
                  enddo
                  isok = matPutFull(fp,'hbzero',1,1,hbzero,0.0d0)
               endif
            endif
c                 
c           Internal MATLAB sparse indices are zero based.
c
            do k = 1, nnz
               ia(k) = ia(k)-1
            enddo
            do k = 1, n+1
               ja(k) = ja(k)-1
            enddo
            ap = mxCreateSparse(m, n, nnz, acmplx)
            call mxSetName(ap,'A')
            call mxSetPr(ap,%loc(a(1)))
            if (acmplx) call mxSetPi(ap,%loc(a(nnz+1)))
            call mxSetIr(ap,%loc(ia(1)))
            call mxSetJc(ap,%loc(ja(1)))
            isok = matPutMatrix(fp, ap)
            call mxFreeMatrix(ap)
         else
c
c           Unassembled, packed matrix
c
            if (aval) then
               ap = mxCreateFull(neltvl, 1, 0)
               call mxSetName(ap,'A')
               call mxSetPr(ap,%loc(a(1)))
               isok = matPutMatrix(fp, ap)
               call mxFreeMatrix(ap)
            endif
            ap = mxCreateFull(1, n+1, 0)
            call mxSetName(ap,'elptr')
            do k = 1, n+1
               a(k) = ja(k)
            enddo
            call mxSetPr(ap,%loc(a(1)))
            isok = matPutMatrix(fp, ap)
            call mxFreeMatrix(ap)
            ap = mxCreateFull(1, nnz, 0)
            call mxSetName(ap,'varind')
            do k = 1, nnz
               a(k) = ia(k)
            enddo
            call mxSetPr(ap,%loc(a(1)))
            isok = matPutMatrix(fp, ap)
            call mxFreeMatrix(ap)
         endif
c
         if (nrhs .gt. 0) then
            mnrhs = m*nrhs
            isok = matPutString(fp, 'rhstype', rhstype)
c
c           Read any right hand sides
c
            if (rhstype(1:1) .eq. 'F') then
c
c              Full right hand sides
c
               read (iounit,rhsfmt) (a(i), i = 1, mnrhs)
               bp = mxCreateFull(m, nrhs, 0)
               call mxSetName(bp,'B')
               call mxSetPr(bp,%loc(a(1)))
               isok = matPutMatrix(fp, bp)
               call mxFreeMatrix(bp)
            else
c
c              Sparse right hand sides
c
               read (iounit,ptrfmt) (ja (i), i = 1, nrhs+1)
               read (iounit,indfmt) (ia (i), i = 1, nnzrhs)
               read (iounit,valfmt) (a(i), i = 1, nnzrhs)
               do k = 1, nnzrhs
                  ia(k) = ia(k)-1
               enddo
               do k = 1, nrhs+1
                  ja(k) = ja(k)-1
               enddo
               bp = mxCreateSparse(m, nrhs, nnzrhs, 0)
               call mxSetName(bp,'B')
               call mxSetPr(bp,%loc(a(1)))
               call mxSetIr(bp,%loc(ia(1)))
               call mxSetJc(bp,%loc(ja(1)))
               isok = matPutMatrix(fp, bp)
               call mxFreeMatrix(bp)
            endif
c
c           Read any starting guesses
c
            if (rhstype(2:2) .eq. 'G') then
               read (iounit,rhsfmt) (a(i), i = 1, mnrhs)
               gp = mxCreateFull(m, nrhs, 0)
               call mxSetName(bp,'G')
               call mxSetPr(gp,%loc(a(1)))
               isok = matPutMatrix(fp, gp)
               call mxFreeMatrix(gp)
            endif
c
c           Read any solution vectors
c
            if (rhstype(3:3) .eq. 'X') then
               read (iounit,rhsfmt) (a(i), i = 1, mnrhs)
               xp = mxCreateFull(m, nrhs, 0)
               call mxSetName(bp,'X')
               call mxSetPr(xp,%loc(a(1)))
               isok = matPutMatrix(fp, xp)
               call mxFreeMatrix(xp)
            endif
         endif
c
         isok = matClose(fp)
      enddo
   99 stop
      end
