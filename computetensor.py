#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: HyperGraph Matching
# Author: Bing Shi
# Institution: Department of Electronic and Engineering, City University of Hong Kong
# Date: 04/03/2017 ~ 03/31/2018
# Version: 4v
# Modify: 
#       affinity: [ (0)[ [s1,[g1b1,g1b2,g2b1,g2b2,Omega],[g1b1,g1b2,g2b1,g2b2,Omega],...,[]], [ s2,[],[],...,[] ],..., [ ]], (1)[ ],..., (n1-1)[ ] ]

''' Note: Data must be cleaned! '''
#Define of data type

#from __future__ import division
#import  shelve
#import  pdb
import  os
import  sys
import  numpy as np
from    math     import  pi, sqrt, exp, fsum
from    datetime import  datetime
from    copy     import  deepcopy

# ////////// Graph ///////// *
# setting of parameter for Graph
# length_Sift=128  #Sift feature

class Graph(object):
    """ """
    __slots__ = ('number', 'keypoints', 'features', 'mname', 'distances', 'neighbours')
    
    def __init__(self,kfn, ffn):
        try:
            self.keypoints = np.loadtxt(kfn, dtype=np.float32)   #load keypoints
        except:
            print('Keypoints read error!')
            exit()
        try:    
            self.features = np.loadtxt(ffn, dtype=np.float32)    #load features
        except:
            print('Features read error!')
            exit()
        self.number = self.keypoints.shape[0]
        if self.features.shape[0] != self.number:
            print("Number of keypoints not equal to number of features!")  
            exit() 
        self.mname=os.path.splitext(os.path.splitext(kfn)[0])[0]
    # end    
        
    def caladistance(self):
        x1, x2 = np.meshgrid(self.keypoints[:, 0], self.keypoints[:, 0])
        y1, y2 = np.meshgrid(self.keypoints[:, 1], self.keypoints[:, 1])
        deltax = x2-x1
        deltay = y2-y1
        self.distances = np.sqrt(deltax**2 + deltay**2)
    # end
    
    def searchneighbor(self):
        ''' note: asymmetry of neighbor relation '''
        self.neighbours = self.distances.argsort()    # indexer of neighbours
    # end
    
    def save(self):      
        np.savetxt(self.mname+'.dist.txt', self.distances, fmt="%f", delimiter=' ', newline='\r\n')
        np.savetxt(self.mname+'.neighbor.txt', self.neighbours, fmt="%u", delimiter=' ', newline='\r\n')
    # end

# End of graph

# ////////// Graph Matching ////////// #

# parameter setting for graphMatch
k = 10                  #number of neighbours to form feature
k2 = 20                 #k~k2-1 neighbours to form hyperedge, adjust to need!!!
simiprop = 0.2
numsimimin = 10         #min of simia number. must be firmly believe that similarity within numsimimin.     
numsimimax = 15         #max of simia number
cutoffrate = 1.4        #similarity of g1 to g2 base on features
thresholdv = 0.000001   #affinity tensor of g1 with g2 base on geometrical relationship
eps = pi/1.5            #kernel bandwidth, reference to Michael et al.:Efficient High Order Matching       
dratio = 0.5            #ratio test
fcdratio = 1.5          #features corresponding deviation ratio to mean
odratio = 3.0           #outliers deviation ratio to mean

class graphMatch(object):
    '''    '''
    __slots__ = ('g1', 'g2', 'alikes', 'likecount', 'fd01', 'afften', 'x_cur', 'x_max', 'x_cur_max',\
                 'K', 'L', 'tabuTab', 'candidate',  'x_sec', 'K_sec','L_sec', 'cmaxv', 'maxv', 'tabuTab2')

    def __init__(self,g1, g2):
        ''' self.alikes = []   self.afften = [] '''
        global k2, numsimimin, numsimimax
        self.g1, self.g2 = g1, g2
        if g1.number > g2.number:
            self.g1, self.g2 = g2, g1         
        if k2 < g1.number/5:                 
            k2 = round(g1.number/5)
        if numsimimin < g2.number * simiprop:        
            numsimimin = round(g2.number * simiprop)
            numsimimax = numsimimin + 5
        if self.g1.number < k2 or self.g2.number < k2:
            print('Graph is very small. that is parameter k2 is large.')
            exit(0)
        print('init: k2=%d, simimin=%d, simimax=%d'%(k2, numsimimin, numsimimax))        
    # end
    
    # Functions: seek similarity in G2 for every point in G1. Firmly believe that similarity within numsimimin.
    def seeksimi(self):
        ''' seek simi of g1 in g2 base on features, ordered base on similarity. '''
        self.alikes = []
        self.fd01 = []       
        for fe1 in self.g1.features:
            simi=[sqrt(fsum((fe1-fe2)**2)) for fe2 in self.g2.features ]       # method to compute distance !!!
            simiarr=np.array(simi)
            indexer=simiarr.argsort()
            ind2 = []
            count = 0    
            for i in indexer[:numsimimax]:      
                if count >= numsimimin and simi[i] > cutoffrate * simi[indexer[0]]:   # Note£º simi is not equal length
                    break
                count += 1
                ind2.append(i)
            self.alikes.append(ind2)
            self.fd01.append([simi[indexer[0]],simi[indexer[1]]])
        # end
        self.likecount = sum([len(item) for item in self.alikes])
        print('likecount: %d'%self.likecount)
        self.savealikes()        #note: every row is not same length!
    # end    
    
    def savealikes(self):
        ''' note: len of every row is not equal to each other '''
        f = open('alikes.txt', 'w')
        for line in self.alikes:
            for item in line:
                f.write(str(item)+' ')
            f.write('\r\n')
        f.close()
    # end
    
    def featcorres(self):
        ''' get partial solutions and get rid of partial outliers
            tabuTab2: -1--unhandle, -2--outlier, 0~n-1--matched '''
        self.tabuTab2 = np.zeros(self.g1.number)
        self.tabuTab2[:] = -1
        fdarr = np.array(self.fd01)
        mean01 = np.mean(fdarr, axis=0)
        std01 = np.std(fdarr, axis=0)
        for i in range(self.g1.number):
            if fdarr[i][0] < dratio*fdarr[i][1]:
                if self.checkhomo(i, self.alikes[i][0]) == True:
                    if fdarr[i, 0] < mean01[0] - fcdratio * std01[0]:        # must be self.alikes[i][0]
                        self.tabuTab2[i] = self.alikes[i][0]                 # need more verify with tensor 
            elif fdarr[i][0] > mean01[0] + odratio * std01[0]:               # !!!
                if self.checkhomo(i, self.alikes[i][0]) == False:
                    self.tabuTab2[i] = -2
        np.savetxt('featcorres.txt',self.tabuTab2,fmt='%d',newline='\r\n')
    # end    
    
    def checkhomo(self, i, j):
        ''' Check homography: i in g1, j in g2 '''
        fe2 = self.g2.features[j]
        simi=[sqrt(fsum((fe1-fe2)**2)) for fe1 in self.g1.features ]         # method to compute distance !!!
        simiarr=np.array(simi)
        indexer=simiarr.argsort()
        if indexer[0] == i:
            if simiarr[indexer[0]] < dratio*simiarr[indexer[1]]:             # ratio test 
                return  True
        return  False
    # end
    
    def calcos(self, c1, c2, c3):
        ''' funct: calculate cos values of three angles of a triangles
            RuntimeWarning: invalid value encountered in true_divide. (if two point is same, ZeroDivisionError. )
            A(a,b) B(c,d) C(e,f)    AB=(c-a,d-b),BC=(e-c,f-d)
            cosABC=AB.BC/|AB|*|BC|=(c-a,d-b)*(e-c,f-d)/|(c-a,d-b)|*|(e-c,f-d)| '''  
        a=[c3[0]-c2[0], c3[1]-c2[1]]      
        b=[c3[0]-c1[0], c3[1]-c1[1]]
        c=[c2[0]-c1[0], c2[1]-c1[1]]
        a0 = sqrt(a[0]**2+a[1]**2)        
        b0 = sqrt(b[0]**2+b[1]**2)
        c0 = sqrt(c[0]**2+c[1]**2)
        assert a0, 'ZeroDivisionError'
        assert b0, 'ZeroDivisionError'
        assert c0, 'ZeroDivisionError'
        a1 = [a[0]/a0, a[1]/a0]          
        b1 = [b[0]/b0, b[1]/b0]          
        c1 = [c[0]/c0, c[1]/c0]          
        cosc1 = b1[0]*c1[0]+b1[1]*c1[1]  
        cosc2 =-a1[0]*c1[0]-a1[1]*c1[1]
        cosc3 = b1[0]*a1[0]+b1[1]*a1[1]
        return (cosc1, cosc2, cosc3)     
    # end
    
    # Save data while computation. For big data. 4th version
    def affinity4(self):
        ''' funct: calculate affinity Tensor '''
        print('begin calculate affinity ...')
        beg = datetime.now()
        f = open('affinity.txt', 'w')                                    # self.afften = []
        g1n = self.g1.neighbours
        g2n = self.g2.neighbours
        g1k = self.g1.keypoints
        g2k = self.g2.keypoints
        for ind_g1c in range(self.g1.number):                            # in subspace
            print('aff: %d'%ind_g1c)
            layer1 = []
            for ind_g2s in self.alikes[ind_g1c]:
                layer2 = [ind_g2s]
                for (i, ind_g1b1) in enumerate(g1n[ind_g1c][k:(k2-1)]):  # three points ordered on neighbor
                    for ind_g1b2  in g1n[ind_g1c][(i+k+1):k2]:
                        u = set(self.alikes[ind_g1b1]) | set(self.alikes[ind_g1b2])
                        for (j, ind_g2b1) in enumerate(g2n[ind_g2s][k:(k2-1)]):
                            if ind_g2b1 not in u: continue
                            for ind_g2b2  in g2n[ind_g2s][(j+k+1):k2]:
                                if ind_g2b2 not in u: continue   
                                (g1t1, g1t2, g1t3) = self.calcos(g1k[ind_g1c], g1k[ind_g1b1], g1k[ind_g1b2])
                                (g2t1, g2t2, g2t3) = self.calcos(g2k[ind_g2s], g2k[ind_g2b1], g2k[ind_g2b2])
                                a = abs(g2t2-g1t2) + abs(g2t3-g1t3)
                                b = abs(g2t2-g1t3) + abs(g2t3-g1t2)
                                if a <= b:
                                    delta = abs(g2t1-g1t1) + a
                                    Omega = exp(-delta/(eps**2))         
                                    if Omega < thresholdv: continue
                                    layer3=(ind_g1b1, ind_g1b2, ind_g2b1, ind_g2b2, Omega)     
                                else:
                                    delta = abs(g2t1-g1t1) + b
                                    Omega = exp(-delta/(eps**2))         
                                    if Omega < thresholdv: continue
                                    layer3=(ind_g1b1, ind_g1b2, ind_g2b2, ind_g2b1, Omega)          
                                f.write(str(ind_g1c)+' ')
                                f.write(str(ind_g2s)+' ')
                                for item in layer3:
                                     f.write(str(item)+' ')
                                f.write('\r\n')
        # end of for      
        f.close()
        end = datetime.now()
        print('end calculate affinity! time-consuming: %f seconds'%(end-beg).seconds)   
        return  
    # end of affinity(self)
  
 # End of graphMatch
 
def main(f1, f2):    
    g1 = Graph(f1+'.kps.1.txt', f1+'.des.1.txt') 
    g2 = Graph(f2+'.kps.1.txt', f2+'.des.1.txt')     
    g1.caladistance()
    g1.searchneighbor()        #g1.save()
    g2.caladistance()
    g2.searchneighbor()        #g2.save()  
    gm = graphMatch(g1, g2)    #precondition: n1 <= n2
    gm.seeksimi()
    gm.featcorres()
    gm.affinity4()
# End
    
if __name__ == "__main__":
    # environment config
    main(sys.argv[1], sys.argv[2])   # main('g1', 'g2')
