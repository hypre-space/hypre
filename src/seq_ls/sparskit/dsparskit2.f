c----------------------------------------------------------------------- 
      integer function maskdeg2 (ja1,ia1,ja2,ia2,nod,mask,maskval) 
      implicit none 
      integer ja1(*),ia1(*),ja2(*),ia2(*),nod,mask(*),maskval
c-----------------------------------------------------------------------
      integer deg, k
      deg = 0
      do k =ia1(nod),ia1(nod+1)-1
         if (mask(ja1(k)) .eq. maskval) deg = deg+1 
      enddo
      do k =ia2(nod),ia2(nod+1)-1
         if( ja2(k).ne.nod) then
            if (mask(ja2(k)) .eq. maskval) deg = deg+1 
         endif
      enddo
      maskdeg2 = deg 
      return
      end 
c----------------------------------------------------------------------- 
c----------------------------------------------------------------------- 
      subroutine perphn2(n,ja1,ia1,ja2,ia2,
     $     init,iperm,mask,maskval,nlev,riord,levels) 
      implicit none
      integer n,ja1(*),ia1(*),ja2(*),ia2(*),
     $     init,iperm(*),mask(*),maskval,
     *     nlev,riord(*),levels(*)
c-----------------------------------------------------------------------
c     finds a pseudo-peripheral node and does a BFS search from it. 
c     Version 2 allows 2 disjoint matrices as input for structure,
c       usually for a separate lower and upper triangle.(AC)
c-----------------------------------------------------------------------
c input: 
c------- 
c n      = row dimension of matrix == number of vertices in graph
c ja1, ia1  = list pointer array for the first adjacency graph
c ja2, ia2  = list pointer array for the second adjacency graph
c nfirst = number of nodes in the first level that is input in riord
c iperm  = integer array indicating in which order to  traverse the graph
c          in order to generate all connected components. 
c          The nodes will be traversed in order iperm(1),....,iperm(n) 
c          Convention: 
c          if iperm(1) .eq. 0 on entry then BFS will traverse the 
c          nodes in the  order 1,2,...,n. 
c 
c riord  = (also an ouput argument). on entry riord contains the labels  
c          of the nfirst nodes that constitute the first level.      
c 
c mask   = array used to indicate whether or not a node should be 
c maskval = value to be checked against for determing whether or
c           not a node is masked. If mask(k) .ne. maskval then
c           node k is not considered. 
c init    = init node in the pseudo-peripheral node algorithm. 
c
c output:
c-------
c init    = actual pseudo-peripherial node found. 
c nlev    = number of levels in the final BFS traversal. 
c riord  = `reverse permutation array'. Contains the labels of the nodes
c          constituting all the levels found, from the first level to
c          the last. 
c levels = pointer array for the level structure. If lev is a level
c          number, and k1=levels(lev),k2=levels(lev+1)-1, then
c          all the nodes of level number lev are:
c          riord(k1),riord(k1+1),...,riord(k2) 
c-----------------------------------------------------------------------
      integer j,nlevp,deg,nfirst,mindeg,nod,maskdeg2
      nlevp = 0 
 1    continue
      riord(1) = init
      nfirst = 1 
      call BFS2(n,ja1,ia1,ja2,ia2,nfirst,iperm,mask,maskval,
     $          riord,levels,nlev)
      if (nlev .gt. nlevp) then 
         mindeg = levels(nlev+1)-1
         do j=levels(nlev),levels(nlev+1)-1
            nod = riord(j) 
            deg = maskdeg2(ja1,ia1,ja2,ia2,nod,mask,maskval)
            if (deg .lt. mindeg) then
               init = nod
               mindeg = deg
            endif 
         enddo
         nlevp = nlev 
         goto 1 
      endif
      return
      end
c-----------------------------------------------------------------------
      subroutine add_lvst2(istart,iend,nlev,riord,ja1,ia1,ja2,ia2,
     $        mask,maskval)
      implicit none
      integer istart, iend, maskval
      integer nlev, riord(*), ja1(*), ia1(*), ja2(*), 
     $        ia2(*), mask(*) 
      integer ir, i, k, j, nod
c---------------------------------------------------------------------- 
c adds one level set to the previous sets. span all nodes of previous 
c set. Uses Mask to mark those already visited. 
c----------------------------------------------------------------------- 
      nod = iend
      do 25 ir = istart+1,iend 
         i = riord(ir)		
         do 24 k=ia1(i),ia1(i+1)-1
            j = ja1(k)
            if (mask(j) .eq. maskval) then
               nod = nod+1 
               mask(j) = 0
               riord(nod) = j
            endif 
 24      continue
         do 26 k=ia2(i),ia2(i+1)-1
            j = ja2(k)
            if (mask(j) .eq. maskval) then
               nod = nod+1 
               mask(j) = 0
               riord(nod) = j
            endif 
 26      continue
 25   continue
      istart = iend 
      iend   = nod 
      return
c-----------------------------------------------------------------------
      end 
c-----------------------------------------------------------------------
      subroutine BFS2(n,ja1,ia1,ja2,ia2,nfirst,iperm,mask,maskval,
     $     riord,levels,
     *     nlev)
      implicit none 
      integer n,ja1(*),ia1(*),ja2(*),ia2(*),nfirst,iperm(n),mask(n),
     $     riord(*),levels(*),
     *     nlev,maskval 
c-----------------------------------------------------------------------
c finds the level-structure (breadth-first-search or CMK) ordering for a
c given sparse matrix. Uses add_lvst2. Allows an set of nodes to be 
c the initial level (instead of just one node). Allows masked nodes.
c-------------------------parameters------------------------------------
c on entry:
c----------
c n      = number of nodes in the graph 
c ja1, ia1  = list pointer array for the first adjacency graph
c ja2, ia2  = list pointer array for the second adjacency graph
c nfirst = number of nodes in the first level that is input in riord
c iperm  = integer array indicating in which order to  traverse the graph
c          in order to generate all connected components. 
c          The nodes will be traversed in order iperm(1),....,iperm(n) 
c          Convention: 
c          if iperm(1) .eq. 0 on entry then BFS will traverse the 
c          nodes in the  order 1,2,...,n. 
c 
c riord  = (also an ouput argument). on entry riord contains the labels  
c          of the nfirst nodes that constitute the first level.      
c 
c mask   = array used to indicate whether or not a node should be 
c          condidered in the graph. see maskval.
c          mask is also used as a marker of  visited nodes. 
c 
c maskval= consider node i only when:  mask(i) .eq. maskval 
c          maskval must be .gt. 0. 
c          thus, to consider all nodes, take mask(1:n) = 1. 
c          maskval=1 (for example) 
c 
c on return
c ---------
c mask   = on return mask is restored to its initial state. 
c riord  = `reverse permutation array'. Contains the labels of the nodes
c          constituting all the levels found, from the first level to
c          the last. 
c levels = pointer array for the level structure. If lev is a level
c          number, and k1=levels(lev),k2=levels(lev+1)-1, then
c          all the nodes of level number lev are:
c          riord(k1),riord(k1+1),...,riord(k2) 
c nlev   = number of levels found
c-----------------------------------------------------------------------
c Notes on possible usage
c-------------------------
c 1. if you want a CMK ordering from a known node, say node init then
c    call BFS with nfirst=1,iperm(1) =0, mask(1:n) =1, maskval =1, 
c    riord(1) = init.
c 2. if you want the RCMK ordering and you have a preferred initial node
c     then use above call followed by reversp(n,riord)
c 3. Similarly to 1, and 2, but you know a good LEVEL SET to start from
c    (nfirst = number if nodes in the level, riord(1:nfirst) contains 
c    the nodes. 
c 4. If you do not know how to select a good initial node in 1 and 2, 
c    then you should use perphn instead. 
c
c-----------------------------------------------------------------------
c     local variables -- 
      integer j, ii, nod, istart, iend 
      logical permut
      permut = (iperm(1) .ne. 0) 
c     
c     start pointer structure to levels 
c     
      nlev   = 0 
c     
c     previous end
c     
      istart = 0 
      ii = 0
c     
c     current end 
c     
      iend = nfirst
c     
c     intialize masks to zero -- except nodes of first level -- 
c     
      do 12 j=1, nfirst 
         mask(riord(j)) = 0 
 12   continue
c-----------------------------------------------------------------------
 13   continue 
c     
 1    nlev = nlev+1
      levels(nlev) = istart + 1
      call add_lvst2 (istart,iend,nlev,riord,ja1,ia1,ja2,ia2,mask,
     $                maskval) 
      if (istart .lt. iend) goto 1
 2    ii = ii+1 
      if (ii .le. n) then
         nod = ii         
         if (permut) nod = iperm(nod)          
         if (mask(nod) .eq. maskval) then
c     
c     start a new level
c
            istart = iend
            iend = iend+1 
            riord(iend) = nod
            mask(nod) = 0
            goto 1
         else 
            goto 2
         endif
      endif
c----------------------------------------------------------------------- 
 3    levels(nlev+1) = iend+1 
      do j=1, iend
         mask(riord(j)) = maskval 
      enddo
      return
c----------------------------------------------------------------------- 
c-----end-of-BFS--------------------------------------------------------
      end
