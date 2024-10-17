from operator import itemgetter
#itemgetter(item) return a callable object that fetches item from its operand using the operandâ€™s __getitem__() method. If multiple items are specified, returns a tuple of lookup values
import numpy as np
import math
from scipy.stats import norm



def fails(list1, list2):
    """returns number of bit errors"""
    return np.sum(np.absolute(list1 - list2))


def bitreversed(num: int, n) -> int:
    """"""
    return int(''.join(reversed(bin(num)[2:].zfill(n))), 2)
#numpy.core.defchararray.zfill(a, width) [source]
#Return the numeric string left-filled with zeros
#int(num,base) method returns an integer object from any number or string.
#The join() method takes all items in an iterable and joins them into one string.


# ------------ transmitting messages -----------------

def gen_messages(msg_length: int):
    """Generates information message of 0s and 1s"""
    # return [random.randint(0, 1) for i in range(msg_length)]
    return np.random.randint(0, 2, size=msg_length, dtype=int)
#numpy.random.randint(low, high=None, size=None, dtype='l')
#Return random integers from low (inclusive) to high (exclusive).


#def transmit_AWGN(msg, symbol_energy=1.0, noise_power=2.0):
def transmit_AWGN(msg, sigma):
    """transmitting encoded message using BPSK through the channel with AWGN
    Ec - the BPSK symbol energy (linear scale)
    N0 - Noise power spectral density (default N0/2 = 1 or sigma^2)"""
    # return [(2 * x - 1) * math.sqrt(Ec) + math.sqrt(N0 / 2) * random.normalvariate(0, 1) for x in msg]
    noise = np.random.standard_normal(size=len(msg))
    return (1 - 2 * msg) + sigma * noise
    #transmitted = (-1 + 2 * msg) * np.sqrt(symbol_energy) + np.sqrt(noise_power / 2) * noise
    #return -1 * transmitted
    #return -(4 * np.sqrt(symbol_energy) / noise_power) * transmitted
#For Ec/No. It should be multiplied by R to get Eb/No

def llr_input(message, symbol_energy=1.0, noise_power=2.0):
    """"""
    # return [-(4 * math.sqrt(symbol_energy) / noise_power) * y for y in message]
    return (4 * np.sqrt(symbol_energy) / noise_power) * message



# ------------ building polar code mask -----------------


def bhattacharyya_count(N: int, design_snr: float):
    # bhattacharya_param = [0.0 for i in range(N)]
    bhattacharya_param = np.zeros(N, dtype=float)
    # snr = pow(10, design_snr / 10)
    snr = np.power(10, design_snr / 10)
    bhattacharya_param[0] = np.exp(-snr)
    for level in range(1, int(np.log2(N)) + 1):
        B = np.power(2, level)
        for j in range(int(B / 2)):
            T = bhattacharya_param[j]
            bhattacharya_param[j] = 2 * T - np.power(T, 2)
            bhattacharya_param[int(B / 2 + j)] = np.power(T, 2)
    return bhattacharya_param


def phi_inv(x: float):
    if (x>12):
        return 0.9861 * x - 2.3152
    elif (x<=12 and x>3.5):
        return x*(0.009005 * x + 0.7694) - 0.9507
    elif (x<=3.5 and x>1):
        return x*(0.062883*x + 0.3678)- 0.1627
    else:
        return x*(0.2202*x + 0.06448)


def dega_construct(N: int, K: int, dsnr_db: float):
    # bhattacharya_param = [0.0 for i in range(N)]
    mllr = np.zeros(N, dtype=float)
    # snr = pow(10, design_snr / 10)
    #dsnr = np.power(10, dsnr_db / 10)
    sigma_sq = 1/(2*K/N*np.power(10,dsnr_db/10))
    mllr[0] = 2/sigma_sq
    #mllr[0] = 4 * K/N * dsnr
    for level in range(1, int(np.log2(N)) + 1):
        B = np.power(2, level)
        for j in range(int(B / 2)):
            T = mllr[j]
            mllr[j] = phi_inv(T)
            mllr[int(B / 2 + j)] = 2 * T
    return mllr

def pe_dega(N: int, K: int, dsnr_db: float):
    # bhattacharya_param = [0.0 for i in range(N)]
    mllr = np.zeros(N, dtype=float)
    pe = np.zeros(N, dtype=float)
    # snr = pow(10, design_snr / 10)
    #dsnr = np.power(10, dsnr_db / 10)
    sigma = np.sqrt(1/(2*K/N*np.power(10,dsnr_db/10)))
    mllr[0] = 2/np.square(sigma)
    #mllr[0] = 4 * K/N * dsnr
    for level in range(1, int(np.log2(N)) + 1):
        B = np.power(2, level)
        for j in range(int(B / 2)):
            T = mllr[j]
            mllr[j] = phi_inv(T)
            mllr[int(B / 2 + j)] = 2 * T
    #mean = 2/np.square(sigma)
    #var = 4/np.square(sigma)
    for ii in range(N):
        #z = (mllr - mean)/np.sqrt(var)
        #pe[ii] = 1/(np.exp(mllr[ii])+1)
        #pe[ii] = 1 - norm.cdf( np.sqrt(mllr[ii]/2) )
        pe[ii] = 0.5 - 0.5 * math.erf( np.sqrt(mllr[ii])/2 )
    return pe

def A(mask, N, K):
    j = 0
    A_set = np.zeros(K, dtype=int)
    for ii in range(N):
        if mask[ii] == 1:
            A_set[j] = bitreversed(ii, int(math.log2(N)))
            j += 1
    A_set = np.sort(A_set)
    return A_set

def countOnes(num:int):
    ones = 0
    binary = bin(num)[2:]
    len_bin = len(binary)
    for i in range(len_bin):
        if binary[i]=='1':
            ones += 1
    return(ones)

def pw_construct(N: int, K: int, dsnr_db: float):
    w = np.zeros(N, dtype=float)
    n = int(np.log2(N))
    for i in range(N):
        wi = 0
        binary = bin(i)[2:].zfill(n)
        for j in range(n):
            wi += int(binary[j])*pow(2,(j*0.25))
        w[i] = wi
    return w


def G_rows_wt(N: int, K: int):
    w = np.zeros(N, dtype=int)
    for i in range(N):
        w[i] = countOnes(i)
    return w
    

def build_mask(N: int, K: int, design_snr=0):
    """Generates mask of polar code
    in mask 0 means frozen bit, 1 means information bit"""
    # each bit has 3 attributes
    # [order, bhattacharyya value, frozen / imformation position]
    # 0 - frozen, 1 - information
    mask = [[i, 0.0, 1] for i in range(N)]
    # Build mask using Bhattacharya values
    #values = G_rows_wt(N, K)
    values = dega_construct(N, K, design_snr)
    #values = bhattacharyya_count(N, design_snr)
    # set bhattacharyya values
    for i in range(N):
        mask[i][1] = values[i]
    # sort channels due to bhattacharyya values
    mask = sorted(mask, key=itemgetter(1), reverse=False)   #DEGA, RM
    #mask = sorted(mask, key=itemgetter(1), reverse=True)    #bhattacharyya
    # set mask[i][2] in 1 for channels with K lowest bhattacharyya values
    for i in range(N-K):
        mask[i][2] = 0
    # sort channels due to order
    mask = sorted(mask, key=itemgetter(0))
    # return positions bits
    return np.array([i[2] for i in mask])

def rm_build_mask(N: int, K: int, design_snr=0):
    """Generates mask of polar code
    in mask 0 means frozen bit, 1 means information bit"""
    # each bit has 3 attributes
    # [order, bhattacharyya value, frozen / imformation position]
    # 0 - frozen, 1 - information
    mask = [[i, 0, 0.0, 1] for i in range(N)]
    # Build mask using Bhattacharya values
    values = G_rows_wt(N, K) # row_wt(i)=2**(wt(bin(i)), value=wt(bin(i))
    values2 = dega_construct(N, K, design_snr)
    #values = bhattacharyya_count(N, design_snr)
    #Bit Error Prob.
    # set bhattacharyya values
    for i in range(N):
        mask[i][1] = values[i]
        mask[i][2] = values2[i]
    # Sort the channels by Bhattacharyya values
    weightCount = np.zeros(int(math.log2(N))+1, dtype=int)
    for i in range(N):
        weightCount[values[i]] += 1
    bitCnt = 0
    k = 0
    while bitCnt + weightCount[k] <= N-K:
        for i in range(N):
            if values[i]==k:
                mask[i][3] = 0
                bitCnt += 1
        k += 1
    mask2 = []
    for i in range(N):
        if mask[i][1] == k:
            mask2.append(mask[i])
    mask2 = sorted(mask2, key=itemgetter(2), reverse=False)   #DEGA
    remainder = (N-K)-bitCnt
    available = weightCount[k]
    for i in range(remainder):
        mask[mask2[i][0]][3] = 0

    rate_profile = np.array([i[3] for i in mask])
    #mask = sorted(mask, key=itemgetter(0))  #sort based on bit-index
    # return positions bits
    #Modify the profile:
    """
    toFreeze = [21]
    toUnfreeze = [18]
    n = int(math.log2(N))
    for i in range(len(toFreeze)):
        #rate_profile[bitreversed(toFreeze[i], n)] = 0
        #rate_profile[bitreversed(toUnfreeze[i], n)] = 1
        rate_profile[toFreeze[i]] = 0
        rate_profile[toUnfreeze[i]] = 1
    """    
    
    return rate_profile

def RAN87_build_mask(N: int, K: int, design_snr=0, a=1.5): #, a=1.5
    """Generates mask of polar code
    in mask 0 means frozen bit, 1 means information bit"""
    # each bit has 3 attributes
    # [order, bhattacharyya value, frozen / imformation position]
    # 0 - frozen, 1 - information
    mask = [[i, 1, 0.0, 0] for i in range(N)]
    # Build mask using Bhattacharya values
    values = G_rows_wt(N, K)
    values2 = pw_construct(N, K, design_snr)
    #values2 = dega_construct(N, K, design_snr)
    #values = bhattacharyya_count(N, design_snr)
    #Bit Error Prob.
    # set bhattacharyya values
    f = int(np.ceil(np.log2(N)*(a-np.abs(a*(K/N-0.5))**2)))
    for i in range(N):
        mask[i][1] = values[i]  #weight
        mask[i][2] = values2[i] #mLLR
    # sort channels due to bhattacharyya values
    weightCount = np.zeros(int(math.log2(N))+1, dtype=int)
    for i in range(N):
        weightCount[values[i]] += 1

        
    mask = sorted(mask, key=itemgetter(2), reverse=False)   #DEGA, RM
    #mask = sorted(mask, key=itemgetter(1), reverse=True)    #bhattacharyya
    # set mask[i][2] in 1 for channels with K lowest bhattacharyya values
    min_wt = mask[N-1][1]
    #for i in range(N-1,N-1-(K+f)-1,-1):
    for i in range(N-1,N-1-(K+f),-1):
        #mask[i][3] = 1
        if mask[i][1] < min_wt :
            min_wt = mask[i][1]
    nf = 0
    #for i in range(N-1,N-1-(K+f)-1,-1):
    for i in range(N-1,N-1-(K+f),-1): #Pre-freezing the positions with w_min
        if mask[i][1] == min_wt :
            mask[i][3] = -1
            nf += 1
    ibit_cnt = 0
    #cnt = 0
    """while nf_cnt < nf:
        if mask[N-K-1-cnt][1] > min_wt:
            mask[N-K-1-cnt][3] = 1
            nf_cnt += 1
        cnt += 1"""
    for i in range(N-1,-1,-1): 
        if mask[i][3] != -1 and ibit_cnt < K:
            mask[i][3] = 1
            ibit_cnt += 1
        elif mask[i][3] == -1:
            mask[i][3] = 0
        elif i< N-1-(K+f):
            break
        
    # sort channels due to order
    mask = sorted(mask, key=itemgetter(0))
    # return positions bits
    return np.array([i[3] for i in mask])
        
         
    
# ------------ SC decoding functions -----------------


    
def lowerconv(upperdecision: int, upperllr: float, lowerllr: float) -> float:
    """PERFORMS IN LOG DOMAIN
    llr = lowerllr * upperllr - - if uppperdecision == 0
    llr = lowerllr / upperllr - - if uppperdecision == 1
    """
    if upperdecision == 0:
        return lowerllr + upperllr
    else:
        return lowerllr - upperllr


def logdomain_sum(x: float, y: float) -> float:
    if x < y:
        return y + np.log(1 + np.exp(x - y))
    else:
        return x + np.log(1 + np.exp(y - x))


def upperconv(llr1: float, llr2: float) -> float:
    """PERFORMS IN LOG DOMAIN
    llr = (llr1 * llr2 + 1) / (llr1 + llr2)"""
    #return logdomain_sum(llr1 + llr2, 0) - logdomain_sum(llr1, llr2)
    return np.sign(llr1)*np.sign(llr2)*min(abs(llr1),abs(llr2))


def logdomain_sum2(x, y):
    return np.array([x[i] + np.log(1 + np.exp(y[i] - x[i])) if x[i] >= y[i]
                     else y[i] + np.log(1 + np.exp(x[i] - y[i]))
                     for i in range(len(x))])

    
def upperconv2(llr1, llr2):
    """PERFORMS IN LOG DOMAIN
    llr = (llr1 * llr2 + 1) / (llr1 + llr2)"""
    return logdomain_sum2(llr1 + llr2, np.zeros(len(llr1))) - logdomain_sum2(llr1, llr2)


# ------------ shortening and puncturing -----------------


def shorten_last_S_bits(N_base: int, N: int, K: int, dsnr: float):
    """"""
    k_first = K
    k_last = N_base - 1
    template = np.append(np.ones(N, dtype=int), np.zeros(N_base - N, dtype=int))
    while True:
        check_first = np.sum(build_mask(N=N_base, K=k_first, design_snr=dsnr) * template)
        check_last = np.sum(build_mask(N=N_base, K=k_last, design_snr=dsnr) * template)

        if check_first == K:
            mask = build_mask(N=N_base, K=k_first, design_snr=dsnr) * template
            break
        elif check_last == K:
            mask = build_mask(N=N_base, K=k_last, design_snr=dsnr) * template
            break
        else:
            k_middle = int((k_first + k_last) / 2)
            check_middle = np.sum(build_mask(N=N_base, K=k_middle, design_snr=dsnr) * template)
            if check_middle == K:
                mask = build_mask(N=N_base, K=k_middle, design_snr=dsnr) * template
                break
            elif check_middle > K:
                k_last = k_middle
            elif check_middle < K:
                k_first = k_middle
    return [mask, template]


def shorten_last(N_base: int, N: int, K: int, dsnr: float):
    """"""
    mask = build_mask(N=N_base, K=K+(N_base-N), design_snr=dsnr)
    template = np.ones(N_base, dtype=int)
    sh = N_base - N
    for i in range(N_base-1, 0, -1):
        if mask[i] == 1:
            template[i] = 0
            mask[i] = 0
            sh -= 1
        if sh == 0:
            break
    return [mask, template]


def puncture_first(N_base: int, N: int, K: int, dsnr: float):
    """"""
    mask = build_mask(N=N_base, K=K, design_snr=dsnr)
    template = np.ones(N_base, dtype=int)
    sh = N_base - N
    for i in range(N_base):
        if mask[i] == 0:
            template[i] = 0
            sh -= 1
        if sh == 0:
            break
    return [mask, template]


def lowcomp_short(N_base: int, N: int, K: int, dsnr: float):
    """"""
    channels = [[i, 0.0] for i in range(N_base)]
    # Build mask using Bhattacharya values
    values = bhattacharyya_count(N_base, dsnr)
    # set bhattacharyya values
    for i in range(N_base):
        channels[i][1] = values[i]
    # sort channels due to bhattacharyya values
    reliability = [i[0] for i in sorted(channels, key=itemgetter(1), reverse=True)]

    bn = [i for i in range(0, N_base, 2)] + [i for i in range(1, N_base, 2)]
    for i in range(0, N_base, 4):
        bn[i + 1], bn[i + 2] = bn[i + 2], bn[i + 1]
    shorten = bn[N:]

    short_reliability = list()
    for r in reliability:
        if r not in shorten:
            short_reliability.append(r)

    frozen = short_reliability[:N - K]

    template = np.ones(N_base, dtype=int)
    mask = np.ones(N_base, dtype=int)
    for i in range(N_base):
        if i in shorten:
            template[i] = 0
            mask[i] = 0
        if i in frozen:
            mask[i] = 0
    return [mask, template]


def lowcomp_puncture(N_base: int, N: int, K: int, dsnr: float):
    """"""
    bn = [i for i in range(0, N_base, 2)] + [i for i in range(1, N_base, 2)]
    for i in range(0, N_base, 4):
        bn[i + 1], bn[i + 2] = bn[i + 2], bn[i + 1]


def build_shorten_template(N_base: int, N: int, K: int, dsnr: float, shorten_method: str):
    """"""
    if shorten_method == shorten_method_01:
        return shorten_last_S_bits(N_base=N_base, N=N, K=K, dsnr=dsnr)
    elif shorten_method == shorten_method_02:
        return shorten_last(N_base=N_base, N=N, K=K, dsnr=dsnr)
    elif shorten_method == shorten_method_03:
        return puncture_first(N_base=N_base, N=N, K=K, dsnr=dsnr)
    elif shorten_method == shorten_method_04:
        return lowcomp_short(N_base=N_base, N=N, K=K, dsnr=dsnr)


# ------------ CRC -----------------


def int_to_binlist(num: int, size: int):
    """"""
    return [int(bit) for bit in bin(num)[2:].zfill(size)]

def build_crc8_table(crc_poly: int):
    """"""
    generator = np.uint16(crc_poly)
    crc8_table = list()
    for div in range(256):
        cur_byte = div #np.uint8(div << 8)
        for bit in range(8):
            #temp1 = np.bitwise_and(cur_byte, np.uint16(0x8000))
            if np.bitwise_and(cur_byte, np.uint8(0x80)) != np.uint8(0x00):
                # cur_byte = np.left_shift(cur_byte, 1)  #
                cur_byte <<= 1
                # cur_byte = np.bitwise_xor(cur_byte, generator)  #
                cur_byte ^= generator
            else:
                # cur_byte = np.left_shift(cur_byte, 1)  #
                cur_byte <<= 1
        crc8_table.append(np.uint8(cur_byte))
    return crc8_table

def crc8_table_method(info, crc_table):
    """"""
    crc = 0
    if info.size%8 != 0:
        pad0 = np.zeros((info.size//8*8+8)-info.size, dtype=np.int8)
        info = np.append(pad0, info)
    # Byte-oriented: Used for packing every 8 bits (one byte), because data are stored in bytes
    coef = np.array([128, 64, 32, 16, 8, 4, 2, 1])  # for easy left shift by 8
    for b in range(0, len(info), 8):
        pos = np.uint8((crc) ^ np.sum(info[b:b+8] * coef))
        crc = crc_table[pos]
    return int_to_binlist(crc, 8)

def build_crc12_table(crc_poly: int):
    """"""
    generator = np.uint16(crc_poly)
    crc12_table = []
    for div in range(256):
        cur_byte = div << 4
        for bit in range(8):
            cur_byte <<= 1
            if cur_byte & 0x1000:
            #if np.bitwise_and(cur_byte, 0x800) != 0x000:
                cur_byte ^= generator
                pass
            continue
        crc12_table.append(cur_byte & 0xfff)
    return crc12_table

def crc12_table_method(info, crc_table):
    """"""
    crc = 0
    # Byte-oriented: Used for packing every 8 bits (one byte), because data are stored in bytes
    if info.size%8 != 0:
        pad0 = np.zeros((info.size//8*8+8)-info.size, dtype=np.int8)
        info = np.append(pad0, info)
    coef = np.array([128, 64, 32, 16, 8, 4, 2, 1])  # for easy left shift by 8
    for b in range(0, len(info), 8):
        """print(np.uint8(crc >> 4))
        print(info[b:b+8])
        print(info[b:b+8] * coef) #operands could not be broadcast together with shapes (4,) (8,) 
        print(np.sum(info[b:b+8] * coef))
        print(np.uint8(crc >> 4) ^ np.sum(info[b:b+8] * coef))"""
        #pos = np.uint8((crc >> 4) ^ np.sum(info[b:b+8] * coef))
        #crc = np.uint16((crc << 8) ^ crc_table[pos])
        pos = ((crc >> 4) & 0xff) ^ np.sum(info[b:b+8] * coef)
        crc = ((crc << 8) & 0xfff) ^ crc_table[pos]
    #print(crc)
    return int_to_binlist(crc, 12)

def build_crc16_table(crc_poly: int):
    """"""
    generator = np.uint16(crc_poly)
    crc16_table = list()
    for div in range(256):
        cur_byte = np.uint16(div << 8)
        for bit in range(8):
            temp1 = np.bitwise_and(cur_byte, np.uint16(0x8000))
            if np.bitwise_and(cur_byte, np.uint16(0x8000)) != np.uint16(0x0000):
                # cur_byte = np.left_shift(cur_byte, 1)  #
                cur_byte <<= 1
                # cur_byte = np.bitwise_xor(cur_byte, generator)  #
                cur_byte ^= generator
            else:
                # cur_byte = np.left_shift(cur_byte, 1)  #
                cur_byte <<= 1
        crc16_table.append(np.uint16(cur_byte))
    return crc16_table


def crc16_table_method(info, crc_table):
    """"""
    crc = 0
    if info.size%8 != 0:
        pad0 = np.zeros((info.size//8*8+8)-info.size, dtype=np.int8)
        info = np.append(pad0, info)
    coef = np.array([128, 64, 32, 16, 8, 4, 2, 1])  # for easy left shift by 8
    for b in range(0, len(info), 8):
        pos = np.uint16((crc >> 8) ^ np.sum(info[b:b+8] * coef))
        crc = np.uint16((crc << 8) ^ crc_table[pos])
    return int_to_binlist(crc, 16)

####PAC########################################

def conv_1bit(in_bit, cur_state, gen): 
    #This function calculates the 1 bit convolutional output during state transition
    g_len = len(gen)    #length of generator 
    g_bit = in_bit * gen[0]        

    for i in range(1,g_len):       
        if gen[i] == 1:
            #print(i-1,len(cur_state))
            #if i-1 > len(cur_state)-1 or i-1 < 0:
                #print("*****cur_state idex is {0} > {1}, g_len={2}".format(i-1,len(cur_state),g_len))
            g_bit = g_bit ^ cur_state[i-1]
    return g_bit


def getNextStateR(cur_state, m):
#This function finds the next state during state transition
    """next_state = [0 for i in range(m)]
    next_state[0] = cur_state[m-1] # extend (the elements), not append
    next_state[1:m] = cur_state[0:m-1] 
    return next_state"""
    return cur_state

def getNextState(in_bit, cur_state, m):
#This function finds the next state during state transition
    #next_state = []
    if in_bit == 0:
        next_state = [0] + cur_state[0:m-1] # extend (the elements), not append
    else:
        next_state = [1] + cur_state[0:m-1]  #np.append([0], cur_state[0:m-1])     
    return next_state
"""
def conv1bit_getNextStates(in_bit, cur_state1, cur_state2, gen1, gen2, bit_flag):
    m1 = len(gen1)-1
    m2 = len(gen2)-1

    g_bit = in_bit       
    for i in range(1,m1+1):       
        if gen1[i] == 1:
            g_bit = g_bit ^ cur_state1[i-1]
    for i in range(1,m2+1):       
        if gen2[i] == 1:
            g_bit = g_bit ^ cur_state2[i-1]


    if bit_flag == 1:
        
        if in_bit == 0:
            next_state2 = [0] + cur_state2[0:m2-1] # extend (the elements), not append
        else:
            next_state2 = [1] + cur_state2[0:m2-1]  #np.append([0], cur_state[0:m-1])
        next_state1 = cur_state1
    else:
        if in_bit == 0:
            next_state1 = [0] + cur_state1[0:m1-1] # extend (the elements), not append
        else:
            next_state1 = [1] + cur_state1[0:m1-1]  #np.append([0], cur_state[0:m-1])     
        next_state2 = cur_state2
    
    return g_bit, next_state1, next_state2
"""

def conv1bit_getNextStates(in_bit, cur_state1, cur_state2, gen1, gen2, bit_flag):
    m1 = len(gen1)-1
    m2 = len(gen2)-1

    g_bit = in_bit       

    if bit_flag == 1:
        for i in range(2,m1+1):       
            if gen1[i] == 1:
                g_bit = g_bit ^ cur_state1[i-1]
        for i in range(1,m2+1):       
            if gen2[i] == 1:
                g_bit = g_bit ^ cur_state2[i-1]
        if in_bit == 0:
            next_state2 = [0] + cur_state2[0:m2-1] # extend (the elements), not append
        else:
            next_state2 = [1] + cur_state2[0:m2-1]  #np.append([0], cur_state[0:m-1])
        if in_bit == 0:
            next_state1 = [0] + cur_state1[0:m1-1] # extend (the elements), not append
        else:
            next_state1 = [1] + cur_state1[0:m1-1]  #np.append([0], cur_state[0:m-1])
        #next_state1 = cur_state1
    else:
        for i in range(1,m1+1):       
            if gen1[i] == 1:
                g_bit = g_bit ^ cur_state1[i-1]
        for i in range(2,m2+1):       
            if gen2[i] == 1:
                g_bit = g_bit ^ cur_state2[i-1]
        if in_bit == 0:
            next_state1 = [0] + cur_state1[0:m1-1] # extend (the elements), not append
        else:
            next_state1 = [1] + cur_state1[0:m1-1]  #np.append([0], cur_state[0:m-1])     
        next_state2 = cur_state2
    
    return g_bit, next_state1, next_state2




"""
def conv_1bit2(in_bit, cur_state1, cur_state2, gen1, gen2, bit_flag):
    m1 = len(gen1)-1
    m2 = len(gen2)-1

    g_bit = in_bit       
    for i in range(1,m1+1):       
        if gen1[i] == 1:
            g_bit = g_bit ^ cur_state1[i-1]
    for i in range(1,m2+1):       
        if gen2[i] == 1:
            g_bit = g_bit ^ cur_state2[i-1]

    return g_bit
"""

def conv_1bit2(in_bit, cur_state1, cur_state2, gen1, gen2, bit_flag):
    m1 = len(gen1)-1
    m2 = len(gen2)-1

    g_bit = in_bit       
    if bit_flag == 1:
        for i in range(2,m1+1):       
            if gen1[i] == 1:
                g_bit = g_bit ^ cur_state1[i-1]
        for i in range(1,m2+1):       
            if gen2[i] == 1:
                g_bit = g_bit ^ cur_state2[i-1]
    else:
        for i in range(1,m1+1):       
            if gen1[i] == 1:
                g_bit = g_bit ^ cur_state1[i-1]
        for i in range(2,m2+1):       
            if gen2[i] == 1:
                g_bit = g_bit ^ cur_state2[i-1]

    return g_bit


def conv_1bit2R(in_bit, cur_state1, cur_state2, gen1, gen2, frozen_flag):
    m1 = len(gen1)-1
    m2 = len(gen2)-1

    g_bit = in_bit       
    #if frozen_flag == 0:
    for i in range(1,m2+1):       
        if gen2[i] == 1:
            g_bit = g_bit ^ cur_state2[i-1]
    for i in range(1,m1+1):       
        if gen1[i] == 1:
            g_bit = g_bit ^ cur_state1[i-1]
    """else:
        for i in range(1,m1+1):       
            if gen1[i] == 1:
                g_bit = g_bit ^ cur_state1[i-1]"""

    return g_bit


def getNextState2(in_bit, cur_state1, cur_state2, m1, m2, bit_flag):

    if bit_flag == 1:
        if in_bit == 0:
            next_state2 = [0] + cur_state2[0:m2-1] # extend (the elements), not append
        else:
            next_state2 = [1] + cur_state2[0:m2-1]  #np.append([0], cur_state[0:m-1])     
        if in_bit == 0:
            next_state1 = [0] + cur_state1[0:m1-1] # extend (the elements), not append
        else:
            next_state1 = [1] + cur_state1[0:m1-1]  #np.append([0], cur_state[0:m-1])  
        #next_state1 = cur_state1
    else:
        if in_bit == 0:
            next_state1 = [0] + cur_state1[0:m1-1] # extend (the elements), not append
        else:
            next_state1 = [1] + cur_state1[0:m1-1]  #np.append([0], cur_state[0:m-1])     
        next_state2 = cur_state2
    
    return next_state1, next_state2


def getNextState2R(in_bit, cur_state1, cur_state2, m1, m2, bit_flag):

    if bit_flag == 1:
        if in_bit == 0:
            next_state2 = [0 ^ cur_state2[m2-1]] + cur_state2[0:m2-1] # extend (the elements), not append
        else:
            next_state2 = [1 ^ cur_state2[m2-1]] + cur_state2[0:m2-1]  #np.append([0], cur_state[0:m-1])     
        if in_bit == 0:
            next_state1 = [0] + cur_state1[0:m1-1] # extend (the elements), not append
        else:
            next_state1 = [1] + cur_state1[0:m1-1]  #np.append([0], cur_state[0:m-1])
        next_state1 = cur_state1
    else:
        if in_bit == 0:
            next_state1 = [0] + cur_state1[0:m1-1] # extend (the elements), not append
        else:
            next_state1 = [1] + cur_state1[0:m1-1]  #np.append([0], cur_state[0:m-1])     
        next_state2 = cur_state2
        # Cycling
        """next_state2 = [0 for i in range(m2)]
        next_state2[0] = cur_state2[m2-1] # extend (the elements), not append
        next_state2[1:m2] = cur_state2[0:m2-1]"""
    
    return next_state1, next_state2

"""
# Combining the previous value of rows with min-weight an sending them on the these bits, the rest are separately combined and sent
def conv1bit_getNextStates(in_bit, cur_state1, cur_state2, gen1, gen2, bit_flag):
    m1 = len(gen1)-1
    m2 = len(gen2)-1

    g_bit = in_bit       


    if bit_flag == 1:
        for i in range(1,m2+1):       
            if gen2[i] == 1:
                g_bit = g_bit ^ cur_state2[i-1]
        if in_bit == 0:
            next_state2 = [0] + cur_state2[0:m2-1] # extend (the elements), not append
        else:
            next_state2 = [1] + cur_state2[0:m2-1]  #np.append([0], cur_state[0:m-1])
        next_state1 = cur_state1
    else:
        for i in range(1,m1+1):       
            if gen1[i] == 1:
                g_bit = g_bit ^ cur_state1[i-1]
        if in_bit == 0:
            next_state1 = [0] + cur_state1[0:m1-1] # extend (the elements), not append
        else:
            next_state1 = [1] + cur_state1[0:m1-1]  #np.append([0], cur_state[0:m-1])     
        next_state2 = cur_state2
    
    return g_bit, next_state1, next_state2

def conv_1bit2(in_bit, cur_state1, cur_state2, gen1, gen2, bit_flag):
    m1 = len(gen1)-1
    m2 = len(gen2)-1

    g_bit = in_bit       
    if bit_flag == 1:
        for i in range(1,m2+1):       
            if gen2[i] == 1:
                g_bit = g_bit ^ cur_state2[i-1]
    else:
        for i in range(1,m1+1):       
            if gen1[i] == 1:
                g_bit = g_bit ^ cur_state1[i-1]

    return g_bit

def getNextState2(in_bit, cur_state1, cur_state2, m1, m2, bit_flag):

    if bit_flag == 1:
        if in_bit == 0:
            next_state2 = [0] + cur_state2[0:m2-1] # extend (the elements), not append
        else:
            next_state2 = [1] + cur_state2[0:m2-1]  #np.append([0], cur_state[0:m-1])     
        next_state1 = cur_state1
    else:
        if in_bit == 0:
            next_state1 = [0] + cur_state1[0:m1-1] # extend (the elements), not append
        else:
            next_state1 = [1] + cur_state1[0:m1-1]  #np.append([0], cur_state[0:m-1])     
        next_state2 = cur_state2
    
    return next_state1, next_state2
"""



def conv_encode(in_code, gen, m):
    # function to find the convolutional code for given input code (input code must be padded with zeros)
    #cur_state = np.zeros(m, dtype=np.int)         # intial state is [0 0 0 ...]
    cur_state = [0 for i in range(m)]#np.zeros(m, dtype=int)
    len_in_code = len(in_code)           # length of input code padded with zeros
    conv_code = np.zeros(len_in_code, dtype=int)     
    log_N = int(math.log2(len_in_code))
    for j in range(0,len_in_code):
        i = bitreversed(j, log_N)
        in_bit = in_code[i]              # 1 bit input 
        #if cur_state.size==0:
            #print("*****cur_state len is {0}, m={1}".format(cur_state.size,m))
        output = conv_1bit(in_bit, cur_state, gen);    # transition to next state and corresponding 2 bit convolution output
        cur_state = getNextState(in_bit, cur_state, m)    # transition to next state and corresponding 2 bit convolution output
        #conv_code = conv_code + [output]  #list   # append the 1 bit output to convolutional code
        conv_code[i] = output
    return conv_code

def convR_encode(in_code, gen, m, pc_mask):
    # function to find the convolutional code for given input code (input code must be padded with zeros)
    #cur_state = np.zeros(m, dtype=np.int)         # intial state is [0 0 0 ...]
    cur_state = [0 for i in range(m)]#np.zeros(m, dtype=int)
    len_in_code = len(in_code)           # length of input code padded with zeros
    conv_code = np.zeros(len_in_code, dtype=int)     
    log_N = int(math.log2(len_in_code))
    for j in range(0,len_in_code):
        i = bitreversed(j, log_N)
        in_bit = in_code[i]              # 1 bit input 
        #if cur_state.size==0:
            #print("*****cur_state len is {0}, m={1}".format(cur_state.size,m))
        if pc_mask[i] == 1:
            output = conv_1bit(in_bit, cur_state, gen);    # transition to next state and corresponding 2 bit convolution output
            cur_state = getNextState(in_bit, cur_state, m)    # transition to next state and corresponding 2 bit convolution output
        else:
            output = 0;    # transition to next state and corresponding 2 bit convolution output
            cur_state = getNextStateR(cur_state, m)    # transition to next state and corresponding 2 bit convolution output
        #conv_code = conv_code + [output]  #list   # append the 1 bit output to convolutional code
        conv_code[i] = output
    return conv_code

def conv2_encode(in_code, gen1, gen2, m1, m2, pc_mask, bit_flag):
    # function to find the convolutional code for given input code (input code must be padded with zeros)
    #cur_state = np.zeros(m, dtype=np.int)         # intial state is [0 0 0 ...]
    cur_state1 = [0 for i in range(m1)]#np.zeros(m, dtype=int)
    cur_state2 = [0 for i in range(m2)]#np.zeros(m, dtype=int)
    len_in_code = len(in_code)           # length of input code padded with zeros
    conv_code = np.zeros(len_in_code, dtype=int)     
    log_N = int(math.log2(len_in_code))
    for j in range(0,len_in_code):
        i = bitreversed(j, log_N)
        in_bit = in_code[i]              # 1 bit input 
        output, cur_state1, cur_state2 =  conv1bit_getNextStates(in_bit, cur_state1, cur_state2, gen1, gen2, bit_flag[j])
        conv_code[i] = output
    return conv_code


def conv2R_encode(in_code, gen1, gen2, m1, m2, pc_mask, bit_flag, frozen_flag):
    # function to find the convolutional code for given input code (input code must be padded with zeros)
    #cur_state = np.zeros(m, dtype=np.int)         # intial state is [0 0 0 ...]
    cur_state1 = [0 for i in range(m1)]   #np.zeros(m, dtype=int)
    cur_state2 = [0 for i in range(m2)]   #np.zeros(m, dtype=int)
    len_in_code = len(in_code)           # length of input code padded with zeros
    conv_code = np.zeros(len_in_code, dtype=int)     
    log_N = int(math.log2(len_in_code))
    for j in range(0,len_in_code):
        i = bitreversed(j, log_N)
        in_bit = in_code[i]              # 1 bit input 
        output = conv_1bit2R(in_bit, cur_state1, cur_state2, gen1, gen2, frozen_flag[j])
        cur_state1, cur_state2 =  getNextState2R(in_bit, cur_state1, cur_state2, m1, m2, bit_flag[j])
        conv_code[i] = output
    return conv_code


def generate_critical_set(frozen_bits:np.int):
    N = frozen_bits.size #array
    n = int(np.log2(N))
    #cnt = 0
    #critical_set = np.zeros(N, dtype=np.int8)
    hw = []
    critical_set = []
    A = -1 * np.ones((n + 1, N), dtype=np.int)    #an extra row for frozen_bits
    for ii in range(N):
        A[-1, bitreversed(ii,n)] = frozen_bits[ii]
    #A[-1, :] = frozen_bits
    for i in range(n-1,-1,-1):
        for j in range(0,np.power(2,(i))):
            A[i, j] = A[i + 1, 2 * j] + A[i + 1, 2 * j + 1]

    for i in range(0,n+1): #levels
        for j in range(0,np.power(2,(i))): #nodes in levels
            if A[i, j] == 0:
                index_1 = j
                index_2 = j
                for k in range(1, n - i+1): #expansion to lower levels
                    index_1 = 2 * index_1 #first bit of rate-1 sub-block in each level
                    index_2 = 2 * index_2 + 1
                    for p in range(index_1, index_2+1):
                        A[i + k, p] = -1 #to avoid considering those nodes again in lower levels
                critical_set.append(index_1)
                #critical_set[cnt] = index_1    #first bit in rate-1 node.
                #cnt += 1
                """
                hw0 = (bin(index_1)[2:].zfill(n)).count('1')
                hw.append(hw0)
    hw_min = min(hw)
    #print(len(hw))
    cs_len = len(critical_set)
    idx = 0
    while idx < cs_len:
        hw1 = (bin(critical_set[idx])[2:].zfill(n)).count('1')
        #print(hw1)
        if hw1 > hw_min:
            #print(idx)
            cs_len -= 1
            critical_set.pop(idx)
            hw.pop(idx)
        else:
            idx += 1
    """       
    critical_set.sort() #reverse = True
    return np.array(critical_set)
    #critical_set = np.sort(critical_set[critical_set != 0])


def bin2dec(binary): 
    decimal = 0
    for i in range(len(binary)): 
        decimal = decimal + binary[i] * pow(2, i) 
    return decimal







