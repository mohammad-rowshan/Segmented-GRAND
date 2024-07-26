# Segmented ORB-GRAND for eBCH ###########################################
#
# Copyright (c) 2023, Mohammad Rowshan and Jinhong Yuan
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, 
# are permitted provided that:
# the source code retains the above copyright notice, and te redistribtuion condition.
# 
# Freely distributed for educational and research purposes
#######################################################################################

import numpy as np
#from numpy import random
from scipy.stats import norm
import sympy as sp
import copy
from channel import channel
from rate_profile import rateprofile
import polar_coding_functions as pcf
#import GaloisField
from GaloisField import X, degree
import csv
import math
import time

n = 7 
N = 2**n-1
t = 3 
q = 2**n
p = [1,0,0,0,1,0,0,1] #for GF(2^7)
K = 106 
snrb_snr = 'SNRb'   # 'SNRb':Eb/N0 or 'SNR':Es/N0
modu = 'BPSK'       # QPSK or BPSK

snrb_snr = 'SNRb'   # 'SNRb':Eb/N0 or 'SNR':Es/N0
modu = 'BPSK'       # QPSK or BPSK
poly = [1,1,0,0,0,1,1,1,1,0,0,1,1,0,1,1,0,1,1,0,0,1] #for (127,106) by default prim_poly = 'D^7+D^3+1', MATLAB: [genpoly,t] = bchgenpoly(127,106) >> t=3

b = [10**2, 10**3, 10**4, 10**5, 10**6]#, 10**7] # maximum queries # abondonment threshold

snr_range = np.arange(3,6,0.5)
err_cnt = 3

H_col_set = [0,1] # Column indices are ordered from the largest to smallest in terms of weight
cmt_on_sim = 'eBCH,Multi-gen:S0,S1'#,rho=0.3,epsilon=0.2, k=synd++2+2+2+2'
odd_en = False #True #

# %% Generator matrix
    
G_poly = np.zeros((K,N), dtype=np.int8)
for i in range(0,K):
    if i+len(poly)-1 > N-1:  # In this case, because it is not a square matrix, it doesn't happen
        j = len(poly) - (i+len(poly)-1 - (K-1))
    else:
        j = len(poly)
    G_poly[i][i:i+len(poly)] = poly[0:j]


#G = np.matmul(G_poly,G_N)%2   #In MATLAB: inv_G = inv(gf(G,1))
G = G_poly
#G_inv = np.linalg.inv(G)%2

# Extension of the G matrix
sum_G_rows = np.sum(G,axis=1) # All the same
if sum_G_rows[0]%2 == 1:
    c1 = np.ones((K,1), dtype=np.int8)
else:
    c1 = np.zeros((K,1), dtype=np.int8)
    
G = np.concatenate((c1, G),1)


# %% Parity check matrix
#G_inv = np.linalg.inv(G)%2 #for square matrices

INF = float('inf') # infinity variable
#GF2 = GaloisField(p) # global GF(2) field

def degree(p):
    """Returns degree of polynomial (highest exponent).
    (slide 3)
    """
    poly = np.poly1d(np.flipud(p))
    return poly.order

def X(i):
    """Create single coefficient polynomial with degree i: X^i
    """
    X = np.zeros(i + 1,dtype=np.int8) # including degree 0
    X[i] = 1
    return X#.astype(int)

def constructGF(p):
    """Construct GF(2^m) based on primitive polynomial p.
    The degree of pi(X) is used to determine m.
    (slide 12)
    Args:
        p: primitive polynomial p to construct the GF with.
        verbose: print information on how the GF is constructed.
    Returns:
        Elements of the GF in polynomial representation.
    """
    elements = []
    m = degree(p)  # Degree of the polynomial

    if m == 1: # special simple case: GF(2)
        elements = [np.array([0]), np.array([1])]
        return elements

    a_high = p[1:m+1] #Except the last element # The highest degree of alpha = the rest of the terms. Ex: a^4=1+a for p = a^4+a+1. We ignore signs and even coeff-elements in the GF(2)

    for i in range(0, 2**m):
        # create exponential representation
        if i == 0:
            exp = np.array([0]) #np.zeros(m)
        else:
            exp = X(i-1)

        poly = exp
        if degree(poly) >= m:
            quotient, remainder = divmod(degree(poly), m)   # modulo m of degree(poly)

            poly = X(remainder)
            for j in range(0, quotient):
                #poly = np.pad(poly, (len(a_high) - poly.size-1, 0), 'constant', constant_values = 0)
                poly = np.polymul(np.flipud(poly), a_high)
                poly = np.flipud(poly)%2

            while degree(poly) >= m:
                poly = np.polyadd(np.flipud(poly), np.flipud(elements[degree(poly) + 1]))%2
                poly = np.flipud(poly)
                poly = poly[:degree(poly)] # Discard the last element (with hiest degree/power)

        # format polynomial (size m)
        poly = poly[:degree(poly) + 1]
        poly = np.pad(poly, (0,m - poly.size), 'constant', constant_values = 0)

        # append to elements list for return
        elements.append(poly) #.astype(int))


    return elements

def element(a,q):
    """Return element that is the same as element a but with an
    exponent within 0 and q-1.
    """
    if a == 0: # zero element doesn't have an exponent
        return int(a)
    exp_a = a - 1 # convert from integer representation to exponent
    exp_a = exp_a % (q - 1) # get exponent within 0 and q-1
    a = exp_a + 1 # convert back to integer representation
    return int(a)

def elementFromExp(exp_a,q):
    """Returns element in integer representation from given exponent
    representation. For the zero element an exponent of +-infinity is
    expected by definition.
    """
    #if exp_a == INF or exp_a == -INF: # zero element is special case
        #return 0
    exp_a = exp_a % (q - 1) # element with exponent within 0 and q-1
    a = exp_a + 1 # convert to integer representation # Index in the GF table
    return int(a)

GF_table = constructGF(p)

H = np.zeros(((N-K),N), dtype=np.int8)
h_i = 0
m = degree(p)
for row in range(1, 2*t,2): # 1, 3, ..., 2^t-1
    for col in range(0,N): 
        GF_idx = elementFromExp(row*col,q) # (a^ti)^ni
        H[h_i*m:(h_i+1)*m, col] = GF_table[GF_idx]
    h_i += 1

# Extension of the H matrix ###############################
r1 = np.ones((1,N), dtype=np.int8)
cNp1 = np.zeros((N-K+1,1), dtype=np.int8)
cNp1[0,0] = 1

H = np.concatenate((r1, H),0)
H = np.concatenate((cNp1,H),1)
N += 1
###########################################################
Ht = H
H = np.transpose(H)

GHt = np.matmul(G, H)%2 #H is actuall Ht here



# %% Segmentation

def supp_row(h):
    #bnry = [int(x) for x in list(bin(n).replace("0b", ""))] #'{0:0b}'.format(n)
    #bnry = [x for x in list(bin(n).replace("0b", ""))]
    #bnry.reverse()
    indices_of_1s = set()
    for x in range(len(h)):    #indices_of_1s = np.where(bnry == 1)
        if h[x]==1:
            indices_of_1s |= {x}
    return indices_of_1s

#row_weights = np.sum(H, axis=0)
#min_wt,min_idx = np.min(row_weights), np.argmin(row_weights)
#max_wt,max_idx = np.max(row_weights), np.argmax(row_weights)

intrvl_cnt = len(H_col_set)

H2 = np.zeros((N,intrvl_cnt), dtype=np.int8)
indx_set = [set() for _ in range(len(H_col_set))]
i = 0
for c in H_col_set:
    indx_set[i] = supp_row(Ht[c,:])
    i += 1

indx_list = [[] for _ in range(len(H_col_set))]
piX = []
indx_intrvl = []
i = 0
pi3 = [[],[]]
end = start = accum = 0
for c in H_col_set:
    if c != H_col_set[-1]:
        indx_list[i] = list(indx_set[i] - indx_set[i+1])
        H2[:,i] = (H[:,i]+H[:,i+1])%2
    else:
        indx_list[i] = list(indx_set[i])
        H2[:,i] = H[:,i]

    for j in range(N):
        if H2[j,i] == 1:
            pi3[i] += [j]

    start = accum
    end = start + len(indx_list[i]) - 1
    accum += len(indx_list[i])
    indx_intrvl += [[start,end]]
    
    indx_list[i].sort()
    piX += indx_list[i]
    i += 1
indx_intrvl = np.array(indx_intrvl)
pi2 = np.zeros(N, dtype=np.int8)
for x in piX:
    pi2[x] = piX.index(x)


# %% Error Pattern Generation

# Generation of bases for segments. This part requires to be executed only once for every received sequence.
def bases_gen(w_L_min,synd): # odd:1/even:0
    num_seg = len(synd) # number of constraints/segments
    bases_w_L_min = [] # it is w_L_min vector
    bases_w_L_min_wt = [] # it is the sum of all elements in w_L_min
    bases_pattern = []
    bases_pattern_wt = []
    bases_nz_loc = []
    # The base vector showing all possible vlaues for f_i where i is the segment index. For instance, [[1,0],[1]] gives the bases {[f_1=1,f_2=1],[f_1=0,f_2=1]}
    fi_set =[[] for i in range(num_seg)] 
    # Alternative coefficients of the integer formation of logistic weight, members of fi_set set
    for j in range(num_seg):
        fi_set[j] += [1]     # 1 is shared in both cases of odd- and even-weight patterns
        if synd[j] == 0:
            fi_set[j] += [0]     # 0 is for no error case when we have even-weight pattern for s_j=0
    # Forming all pattern bases: all bases of w_L_min = [w_L_min[0]]*f1, w_L_min[1]]*f2, w_L_min[2]]*f3] along with the corrsponding weight of each bases
    for f0 in fi_set[0]:
        base0_w_L_min = [w_L_min[0]*f0]
        for f1 in fi_set[1]:
            base1_w_L_min = [w_L_min[1]*f1]
            if num_seg==3:
                for f2 in fi_set[2]:
                    base2_w_L_min = [w_L_min[2]*f2]
                    bases_w_L_min += [base0_w_L_min+base1_w_L_min+base2_w_L_min]
                    bases_pattern += [[f0,f1,f2]]
                    bases_w_L_min_wt += [sum(bases_w_L_min[-1])]    # THE WEIGHT OF EACH BASE
            else:
                bases_w_L_min += [base0_w_L_min+base1_w_L_min]
                bases_pattern += [[f0,f1]]
                bases_w_L_min_wt += [sum(bases_w_L_min[-1])]
            bases_pattern_wt += [sum(bases_pattern[-1])]
            base_nz_loc = []
            for j in range(num_seg):
                if bases_pattern[-1][j] == 1:
                    base_nz_loc += [j]
            bases_nz_loc += [base_nz_loc]        
                
    bases_order = np.argsort(np.array(bases_w_L_min_wt)) # Putting the bases in ascebding order based on their weights
        
    order_min = 0 if bases_w_L_min_wt[bases_order[0]] > 0 else 1    #[bases_w_L_min_wt[bases_order[1]],bases_order[0]]
    
    return [bases_w_L_min, bases_w_L_min_wt, bases_pattern_wt, bases_nz_loc, bases_order, order_min]


# Level-1 sub-weights generator based on w_L, repetition and permutation of parts (pts) are allowed. Also, an integer part can be zero (for even number of error in the segment) provided s=0.
def parts_rep(n,w_L_min,synd, bases_w_L_min, bases_w_L_min_wt, bases_pattern_wt, bases_nz_loc, bases_order, order_min): # odd:1/even:0
    pts = []
    current_order = bases_order[order_min]
    while n >= bases_w_L_min_wt[current_order]: # and order_min < len(bases_order):
        
        base_nz_loc = bases_nz_loc[current_order]
        # Generation of all paterns with w_L = n. See Fig. 11 and Alg. 1 in https://arxiv.org/pdf/2305.14892
        # Here we have assumed the maximum of 3 segments, that is why we have max pattern_wt == 3
        # We start from p = w_L_min pattern and adjust their elements from the last element.
        if bases_pattern_wt[current_order] == 1:
            p = copy.deepcopy(bases_w_L_min[current_order])
            c1 = n - bases_w_L_min_wt[current_order]
            p[base_nz_loc[0]] = p[base_nz_loc[0]] + c1
            pts += [p]
        elif bases_pattern_wt[current_order] == 2:
            p = copy.deepcopy(bases_w_L_min[current_order])
            c1 = n - bases_w_L_min_wt[current_order]
            c2 = 0
            p[base_nz_loc[0]] = p[base_nz_loc[0]] + c1
            p[base_nz_loc[1]] = p[base_nz_loc[1]] + c2
            pts += [p]
            while p[base_nz_loc[0]] > bases_w_L_min[current_order][base_nz_loc[0]]:
                c1 -= 1
                c2 += 1
                p = copy.deepcopy(bases_w_L_min[current_order])
                p[base_nz_loc[0]] = p[base_nz_loc[0]] + c1
                p[base_nz_loc[1]] = p[base_nz_loc[1]] + c2
                pts += [p]
        elif bases_pattern_wt[current_order] == 3: # will not be used for 2 segments, like for eBCH(128,106)
            base = copy.deepcopy(bases_w_L_min[current_order])
            p = copy.deepcopy(base)
            c1 = n - bases_w_L_min_wt[current_order]
            c2 = 0
            c3 = 0
            p[base_nz_loc[0]] = p[base_nz_loc[0]] + c1
            p[base_nz_loc[1]] = p[base_nz_loc[1]] + c2
            p[base_nz_loc[2]] = p[base_nz_loc[2]] + c3
            pts += [p]
            while c1 > 0:
                while p[base_nz_loc[0]] > bases_w_L_min[current_order][base_nz_loc[0]]:
                    c1 -= 1
                    c2 += 1
                    p = copy.deepcopy(base)
                    p[base_nz_loc[0]] = p[base_nz_loc[0]] + c1
                    p[base_nz_loc[1]] = p[base_nz_loc[1]] + c2
                    pts += [p]
                base[base_nz_loc[2]] += 1
                c3 += 1
                c2 =0
                c1 = n - (bases_w_L_min_wt[current_order] + c3)
                p = copy.deepcopy(base)
                p[base_nz_loc[0]] = p[base_nz_loc[0]] + c1
                p[base_nz_loc[1]] = p[base_nz_loc[1]] + c2
                #p[base_nz_loc[2]] = p[base_nz_loc[2]] + c3
                pts += [p]
                    
        order_min += 1
        if order_min == len(bases_order):
            break
        current_order = bases_order[order_min]
    return pts

# Level-2 sub-pattern generator (up to weight k parts) based on sub-weight n, integer parts are distinct and non-zero, with maximum value of max_part
def parts_dist(n,k,max_part): # odd:1/even:0
    pts = []
    p = [i for i in range(1,k)]
    p += [n-sum(p)]
    if p[-1] <= p[-2]:
        
        return pts
    if p[-1] <= max_part:
        pts += [p]
    fwd_dir = 1
    while True:
        for i in range(1,k):    #(k-2,-1,-1):
            if i == 1:
                #minus = 1
                rhs = p[k-1] -  1
            else:
                #minus = int(p[k-1-i] + i*(i+1)/2 - sum(p[k-i:k-1])) # in Python, k-1, otherwise k-2
                rhs = int( n - (i*p[k-1-i] + i*(i+1)/2) - sum(p[0:k-1-i]))
            if p[k-1-i]+i < rhs: #p[k-1] - minus:
                p_tmp = p[0:k-1-i] + [i for i in range(p[k-1-i]+1,p[k-1-i]+i+1)]
                p = p_tmp +[n-sum(p_tmp)]
                if p[-1]<max_part:
                    pts += [p]
                fwd_dir = 1
                break
            else:
                fwd_dir = 0
        if fwd_dir == 0 and i == k-1:
            break
    return pts


def parts_distinct(n,synd,max_part):
    S=[]

    if n == 0:
        S += [[]]
        return S
    if synd == 1 and n <= max_part:
        S += [[n]]
    
    k = synd+2  # logistic weight, for odd or even, incremented by 2

    lenS0 = 1 # for any number
    while lenS0 > 0 and k < max_part: # define a bound instead of max_part
        S0 = parts_dist(n,k,max_part)
        lenS0 = len(S0)
        S += S0
        k = k + 2  # logistic weight, for odd or even, incremented by 2
 
    return S



# %% Simulator
class BERFER():
    def __init__(self): # structure that keeps results of BER and FER tests
        self.snr = list()
        self.ber = list()
        self.fer = list()
        self.cplx = list()
        self.cplx_bit = list()
        self.avg_dec_time = list()
        self.avg_query_time = list()
        self.avg_w_L = list()
        self.min_w_L = list()
        self.max_w_L = list()
result = BERFER()


print("({},{}) b={}".format(N, K, b))
print(cmt_on_sim,end=',H_col_set=')
print(H_col_set)
print("BER & BLER & QUERIES evaluation is started\n")

iter_rv = []

# CONSTNATS:
W_L_MIN_INIT = [3,1] #[4,1]  # CONSTANT: minimum logistic weight for odd segment, W_L_MIN_INIT[1], and even segment, W_L_MIN_INIT[0], respectively.
NUM_SEG = len(H_col_set)   # number of constraints
        
# Timer starts
starttime = time.time()
lasttime = starttime

for snr in snr_range:
    print("SNR={} dB".format(snr))
    t = -1
    fer = np.zeros(len(b), dtype=int)
    ber = np.zeros(len(b), dtype=int)
    cplx = np.zeros(len(b), dtype=int)

    sum_w_L = 0
    min_w_L = 0
    max_w_L = 0

    ch = channel(modu, snr, snrb_snr, (K / N))
    
    err_distrib_seg = np.zeros(N, dtype=int)
    low_llr_distrib_seg = np.zeros(N, dtype=int)
    np.random.seed(1000)
    while fer[len(b)-1] < err_cnt: #t < 0: #
        t += 1
        d = np.random.randint(0, 2, size=K, dtype=np.int8)
        
        x0 = np.matmul(d,G)%2

        #x1 = np.matmul(G_inv,x0)%2
        
        modulated_x = ch.modulate(x0)
        y = ch.add_noise(modulated_x)
        
        sllr = ch.calc_llr(y)
        llr = abs(ch.calc_llr(y))
        
        # Sorting: bitonic sorter n.log^2(n), parallel time: log^2(n). pi3 includes the indices of each segment
        pi = [[],[]]
        pi[0] = np.argsort(llr[pi3[0]])
        pi[1] = np.argsort(llr[pi3[1]])
        

       
        teta_y = ch.demodulate(y)
        
        # additional info about the errors in the segments
        e_actual = np.array(((teta_y + x0)%2).tolist())
        e_wt = sum(e_actual)
        e_pos = list()
        e_pos_inv = list()
        e_pos_seg = [[],[]]
        e_pos_seg_inv = [[],[]]

        for ei in range(N):
            if e_actual[ei] == 1:
                if pi2[ei]>2**(n-1)-1:
                    e_pos_seg[1] += [ei]
                    e_pos_seg_inv[1] += [pi2[ei]-2**(n-1)]
                else:
                    e_pos_seg[0] += [ei]
                    e_pos_seg_inv[0] += [pi2[ei]-0]
                e_pos.append(ei)
                err_distrib_seg[pi2[ei]] += 1
 
# %% 
        is_odd = 0
        e_loc_min = np.zeros(N-K, dtype=np.int8)
        e_loc_max = np.zeros(N-K, dtype=np.int8)
        
        g = 0   # Number of queries so far
        cnt_T = 0
        z = np.zeros(N, dtype=np.int8)
        
        synd0 = False
        
        # SED metric, for comparison purposes only.
        #metric = np.sum((np.subtract(y, ch.modulate(np.add(teta_y, z)%2)))**2)
        #metric0 = metric
        #for j in e_pos:
            #metric0 = metric0 - (np.subtract(y[j-1], 1-2*(np.add(teta_y[j-1], 0)%2)))**2 + (np.subtract(y[j-1], 1-2*(np.add(teta_y[j-1], 1)%2)))**2
        ##print("correct_metric=",metric0)
        
        syndrome = np.matmul(np.add(teta_y, z), H) % 2 # H is actually H_transpose
        syndrome2 = np.matmul(np.add(teta_y, z), H2) % 2
        ##print("synd=",syndrome)
        w_L = sum(syndrome2) if sum(syndrome2) > 0 else 1 # minimum/lower-bound for overall logistic weight w_L
        if sum(syndrome) == 0:
            e0 = z
            synd0 = True
        else:
            w_L_min = []                  # starting logistic weight for each segment
            for idx in range(NUM_SEG):
                w_L_min += [W_L_MIN_INIT[syndrome2[idx]]]     # assigning minimum w_L to each segment, corresponding to the syndrome symbols
            max_part = [2**(n-1),2**(n-1)]  # maximum/upper-bound w_L for each segment

            [bases_w_L_min, bases_w_L_min_wt, bases_pattern_wt, bases_nz_loc, bases_order, order_min] = bases_gen(w_L_min, syndrome2)
            
            sum_synd = 1 # initial summation of syndrome elements
            seg_parts = [[],[]]
            while g < b[len(b)-1] and sum_synd > 0:
                # Level-1 integer partitioing where repeated weights are allowed
                seg_order_set = parts_rep(w_L, w_L_min, syndrome2, bases_w_L_min, bases_w_L_min_wt, bases_pattern_wt, bases_nz_loc, bases_order, order_min)   
                
                for seg_order in seg_order_set:
                    
                    # Generate e sequences based on integer partitions of each segment
                    for idx in range(NUM_SEG): # int partitions for each segment based on w_L^(idx)
                         seg_parts[idx] = parts_distinct(seg_order[idx], syndrome2[idx], max_part[idx])
                    # find the de-permuted error cordinate(s) of segment 1 and collect them in e11
                    for seg1_parts in seg_parts[1]:
                        e11 =[]
                        for bit_order in seg1_parts:
                            e11 += [pi3[1][pi[1][bit_order-1]]]
                        # find the de-permuted error cordinate(s) of segment 0 and collect them in e10
                        for seg0_parts in seg_parts[0]:
                            e10 =[]
                            for bit_order in seg0_parts:
                                e10 += [pi3[0][pi[0][bit_order-1]]]
                                
                            e0 = np.zeros(N, dtype=np.int8)
                            # Flip coordinates in the Test Error Pattern: e1
                            e1 = e10 + e11
                            e1_seg = [e10, e11] # flip coordinates grouped in segments. The coordinates are with respect to the codeword-length 
                            e1_seg_order = [seg0_parts, seg1_parts] # order/index of flip cooridnates in each segment. The coordinate orders are with respect to the segment-length 
                            
                            # Euclidean distance:
                            #e_metric = metric
                            #for j in e1:
                            #    e_metric = e_metric - (np.subtract(y[j], 1-2*(np.add(teta_y[j], 0)%2)))**2 + (np.subtract(y[j], 1-2*(np.add(teta_y[j], 1)%2)))**2
                            
                            #print(w_L, seg_order, g, e1_seg, e_metric)
                            ##print("{};{};{};{};{}".format(w_L, seg_order, g, e1_seg, e_metric))
                            g += 1
                            syndrome0 = copy.deepcopy(syndrome)
                            # Test Error Pattern: e0
                            for j in e1:
                                e0[j] = 1
                                # Row-wise computation of syndrome: 
                                syndrome0 = (syndrome0 + H[j][:])%2
                            if sum(syndrome0) == 0:
                                sum_synd = 0
                                synd0 = True
                                break
                # Checking whether we have reached the maximum iteration or Syndrome is zero: if yes, terminate the the process
                            if g == b[len(b)-1]: 
                                break
                        if sum_synd == 0 or g == b[len(b)-1]: 
                            break
                    if sum_synd == 0 or g == b[len(b)-1]: 
                        break
                w_L += 1    # When all error patterns with a specific w_L are generated, w_L is incremented for the next round
                if sum_synd == 0 or g == b[len(b)-1]: 
                    break
              
            #if g == b[len(b)-1]: 
                #break

# %%    
        sum_w_L += w_L
        max_w_L = w_L if w_L > max_w_L else max_w_L
        min_w_L = w_L if w_L < min_w_L else min_w_L
        synd0 = False         
        c = np.add(teta_y, e0)%2
        iter_rv += [g]
        for i in range(len(b)):
            if g <= b[i]:
                for j in range(0, i):
                    cplx[j] += b[j]
                for j in range(i, len(b)):
                    cplx[j] += g #cnt_T
                if not np.array_equal(x0, c):
                    for j in range(0, len(b)):
                        fer[j] += 1
                        ber[j] += pcf.fails(x0, c)
                    print("Error # {0} t={1}, BLER={2:0.2e}, BER={3:0.2e} avg_Queries={4} *********************".format(fer[len(b)-1],t, fer[len(b)-1]/(t+1), ber[len(b)-1]/(N*(t+1)), int(cplx[len(b)-1] / (t+1))))
                else:
                    for j in range(0, i):
                        fer[j] += 1
                        ber[j] += pcf.fails(x0, c)
                break
        if t%10000==0:
            print("@ t={} BLER = {}, avg_Queries={}".format(t, fer/(t+1), cplx / (t+1)))
        #fer += not np.array_equal(message, decoded)
    result.snr.append(snr)
    result.ber.append(ber / ((t + 1) * K))
    result.fer.append(fer / (t + 1))
    result.cplx.append(cplx / (t + 1))
    result.cplx_bit.append(cplx / (t + 1)/N)
    result.avg_w_L.append(sum_w_L/ (t + 1))
    result.min_w_L.append(min_w_L)
    result.max_w_L.append(max_w_L)

    print("\n####################################################\n")
    print("({},{}) b={} ".format(N, K, b),end='')
    print(cmt_on_sim)
    print("SNR={}".format(result.snr))
    print("BLER={}".format(result.fer))
    print("BER={}".format(result.ber))
    print("Queries={}".format(result.cplx))
    print("Queries_bit={}".format(result.cplx_bit))
    print("\n####################################################\n")


    # The current lap-time
    laptime = round((time.time() - lasttime), 2)

    # Total time elapsed since the timer started
    totaltime = round((time.time() - starttime), 2)

    result.avg_dec_time.append( laptime/ (t + 1) )
    result.avg_query_time.append( laptime/ cplx[len(b)-1] )
    
    lasttime = time.time()
    print("Total Time: "+str(totaltime)+" s")
    print("Lap Time: "+str(laptime)+" s")
    print("Time per decoding: "+str(laptime/ (t + 1))+" s")
    print("Avg decoding time={}".format(result.avg_dec_time))
    print("Avg query time={}".format(result.avg_query_time))
    print("\n####################################################\n")
    print("avg w_L={}".format(result.avg_w_L))
    print("min w_L={}".format(result.min_w_L))
    print("max w_L={}".format(result.max_w_L))
    print("\n####################################################\n")

