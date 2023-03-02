import numpy as np
from scipy.sparse import csr_matrix

def regionadjacency(seg):
    '''seg - A region segmented image, such as might be produced by a
             graph cut or superpixel algorithm.  All pixels in each
             region are labeled by an integer.'''
             
    rows, cols = seg.shape
    
    # Identify the unique labels in the image, excluding 0 as a label.
    labels = np.unique(seg)

    N = max(labels)
    
    # Allocate vectors for forming row, col, value triplets used to construct sparse matrix.
    # Forming these vectors first is faster than filling entries directly into the sparse matrix.
    i = [0 for _ in range((3+4*(cols-2))*(rows-1))]
    j = [0 for _ in range((3+4*(cols-2))*(rows-1))]
    s = [0 for _ in range((3+4*(cols-2))*(rows-1))]

    n=0; 
    for r in range(0, rows-1):
        # Handle pixels in 1st column
        i[n] = seg[r,0]
        j[n] = seg[r,1]
        s[n] = 1
        n=n+1
        
        i[n] = seg[r,0]
        j[n] = seg[r+1,0]
        s[n] = 1
        n=n+1
        
        i[n] = seg[r,0]
        j[n] = seg[r+1,1]
        s[n] = 1
        n=n+1
        # now the rest of the column
        for c in range(1, cols-1):
            i[n] = seg[r,c]
            j[n] = seg[r,c+1]
            s[n] = 1
            n=n+1
            
            i[n] = seg[r,c]
            j[n] = seg[r+1,c-1]
            s[n] = 1
            n=n+1
            
            i[n] = seg[r,c]
            j[n] = seg[r+1,c]
            s[n] = 1
            n=n+1
            
            i[n] = seg[r,c]
            j[n] = seg[r+1,c+1]
            s[n] = 1
            n=n+1
            
    # Form the logical sparse adjacency matrix.        
    Am = csr_matrix((s, (i, j)), dtype=bool)
    Am = Am.todense()
    
    # Zero out the diagonal.  
    for r in range(0, N):
        Am[r,r] = 0
                
    # Ensure connectivity both ways for all regions.      
    AmC = np.conjugate(Am)
    AmT = np.array(AmC, ndmin=2, dtype=bool).T
    Am  = np.logical_or(AmT, Am)
                
    '''Am - An adjacency matrix indicating which labeled regions are
            adjacent to each other, that is, they share boundaries. Am
            is sparse to save memory.'''
    return Am

def makeStruct(seg, labimg):
    # Find the average lab value of each superpixel
    imData = [0 for _ in range(9)]
    
    for i in range(0, 3):
        imData[i] = [0 for _ in range(np.amax(seg)+1)]
    
    rows, cols = seg.shape
    Y, X = np.mgrid[1:rows+1, 1:cols+1]
    
    for i in range(0, np.amax(seg)+1):
        mask = (seg==i)
        nm = np.sum(mask)
        imL = labimg[:,:,0]
        imA = labimg[:,:,1]
        imB = labimg[:,:,2]
        imData[0][i] = np.sum(imL[mask])/nm    
        imData[1][i] = np.sum(imA[mask])/nm 
        imData[2][i] = np.sum(imB[mask])/nm 
        
    return imData