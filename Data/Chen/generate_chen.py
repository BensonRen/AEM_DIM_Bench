import numpy as np
import time
import sys
import os
import random
import warnings
#from pytorch_env import *
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy.polynomial.chebyshev as cheb
import numpy.polynomial.legendre as legend
from numpy.testing import assert_equal,assert_almost_equal,assert_raises
import pickle

#FOLDER_PATH = 'C:\\Users\\Lab User\\Desktop\\Ashwin\\MM_Bench\\Data\\Chen'
FOLDER_PATH = '/home/sr365/MM_Bench/Data/Chen'
#from numba import jit

def cheb_fitcurve( x,y,order ):
    x = cheb.chebpts2(len(x))
    order = 64
    coef = legend.legfit(x, y, order);    assert_equal(len(coef), order+1)
    y1 = legend.legval(x, coef)
    err_1 = np.linalg.norm(y1-y) / np.linalg.norm(y)

    coef = cheb.chebfit(x, y, order);    assert_equal(len(coef), order + 1)
    thrsh = abs(coef[0]/1000)
    for i in range(len(coef)):
        if abs(coef[i])<thrsh:
            coef = coef[0:i+1]
            break

    y2 = cheb.chebval(x, coef)
    err_2 = np.linalg.norm(y2 - y) / np.linalg.norm(y)

    plt.plot(x, y2, '.')
    plt.plot(x, y, '-')
    plt.title("nPt={} order={} err_cheby={:.6g} err_legend={:.6g}".format(len(x),order,err_2,err_1))
    plt.show()
    assert_almost_equal(cheb.chebval(x, coef), y)
    #
    return coef

class DefaultConfig(object):
    def parse(self, kwargs):
        '''
        根据字典kwargs 更新 config参数
        '''
        # if len(kwargs)>0:
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        print('user config:')
        #for k, v in self.__class__.__dict__.items():
        for k, v in self.__dict__.items():
            if not k.startswith('__'):
                print("\t{}:    {}".format(k, getattr(self, k)))


    def Init_lenda_tic(self ):
        l_0 = self.lenda_0

        self.lenda_tic=[]
        if self.tic_type=="cheb_":
            pt_1 = cheb.chebpts1(256)
            pt_2 = cheb.chebpts2(256)
            scale = (self.lenda_1-l_0)/2
            off = l_0+scale
            self.lenda_tic = [i * scale + off for i in pt_2]
            assert(self.lenda_tic[0]==l_0 and self.lenda_tic[-1]==self.lenda_1)
        else:

            if( self.lenda_0<500 ):
                self.lenda_tic = list(range(self.lenda_0,500,1))
                l_0 = 500
            self.lenda_tic.extend( list(range(l_0,self.lenda_1+self.lenda_step,self.lenda_step)) )

    def __init__(self,fix_seed=None):
        self.env = 'default' # visdom 环境
        #self.device = pytorch_env(fix_seed)
        self.cnn_model = 'ResNet34' # 使用的模型，名字必须与models/__init__.py中的名字一致
        self.use_gpu = True

        self.nLayer = 10
        #self.nLayer = 20 确实很差，需要进一步改进网络    10/27/2018

        #self.xita = 0
        #self.xita = 0  #angle of incidence
        self.xita = 0

        self.model = 'v1'
        self.env_title = ''
        self.polar = 0

        #########################
        # Normal geometry bound #
        #########################
        self.thick_0 = 5
        self.thick_1 = 50

        ################################
        # Extended geometry bound (50%)#
        ################################
        #self.thick_0 = 5
        #self.thick_1 = 50 + 22.5

        self.tic_type = "cheb_"
        # self.tic_type = "cheb_2"
        self.lenda_0 = 240      #240
        # self.lenda_1 = 260
        self.lenda_1 = 2000
        self.lenda_step = 5
        self.Init_lenda_tic()

#xlabel, ylabel = 'lenda (nm)', ['Reflectivity', 'Transmissivity', 'Absorptivity']
        self.noFeat4CMM = 2     #  CMM用来计算3个系数 0,1,2   三条曲线
        self.normal_Y = 0       #反而不行，需要对MLP重新认识

        self.dump_NN = True
        self.n_dict = None
        self.user_loss = None
        self.fix_graphene_thick=True        #always 0.35nm
        #self.use_bn = "bn"
        self.use_bn = "adaptive"
        #self.use_bn = "none"
        self.loss_curve_title = "Loss"
        #self.loss_curve_title = "Other Loss"

class N_Dict(object):
    map2 = {}

    def __init__(self,config=None):
        self.dicts = {}
        self.config = config
        return

    def InitMap2(self, maters, lendas):
        for mater in maters:
            for lenda in lendas:
                self.map2[mater, lenda] = self.Get(mater, lenda, isInterPolate=True)

    def Load(self, material, path, scale=1):
        df = pd.read_csv(path, delimiter="\t", header=None, names=['lenda', 're', 'im'], na_values=0).fillna(0)
        if scale is not 1:
            df['lenda'] = df['lenda'] * scale
            df['lenda'] = df['lenda'].astype(int)
        self.dicts[material] = df
        rows, columns = df.shape
        # if columns==3:
        print("{}@@@{} shape={}\n{}".format(material, path, df.shape, df.head()))

    def Get(self, material, lenda, isInterPolate=False):
        # lenda = 1547
        n = 1 + 0j
        if material == "air":
            return n
        if material == "Si3N4":
            if self.config is None or self.config.model=='v1':
                return 2. + 0j
            else:
                return 2.46 + 0j
            #return 2.0
        assert self.dicts.get(material) is not None
        df = self.dicts[material]
        assert df is not None
        pos = df[df['lenda'] == lenda].index.tolist()
        # assert len(pos)>=1
        if len(pos) == 0:
            if isInterPolate:
                A = df['lenda'].values  #CHANGE BY A.M. df.as_matrix(columns=['lenda'])
                idx = (np.abs(A - lenda)).argmin()
                if idx == 0:
                    lenda_1, re_1, im_1 = df['lenda'].loc[idx], df['re'].loc[idx], df['im'].loc[idx]
                else:
                    lenda_1, re_1, im_1 = df['lenda'].loc[idx - 1], df['re'].loc[idx - 1], df['im'].loc[idx - 1]
                lenda_2, re_2, im_2 = df['lenda'].loc[idx], df['re'].loc[idx], df['im'].loc[idx]
                re = np.interp(lenda, [lenda_1, lenda_2], [re_1, re_2])
                im = np.interp(lenda, [lenda_1, lenda_2], [im_1, im_2])
            else:
                return None
        elif len(pos) > 1:
            re, im = df['re'].loc[pos[0]], df['im'].loc[pos[0]]
        else:
            re, im = df['re'].loc[pos[0]], df['im'].loc[pos[0]]
        n = re + im * 1j
        return n

class Thin_Film_Filters(object):
    def parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        print('user config:')
        # for k, v in self.__class__.__dict__.items():
        for k, v in self.__dict__.items():
            if not k.startswith('__'):
                print("\t{}:    {}".format(k, getattr(self, k)))

    def __init__(self, layers):
        self.layers = layers

def GraSi3N4_init(r_seed,model, config=None):
    np.set_printoptions(linewidth=np.inf)
    random.seed(r_seed - 1)
    np.random.seed(r_seed)
    if config is None:
        config = DefaultConfig(42)
    else:
        config = config
    if config.model!=model:
        print("\n!!!GraSi3N4_init model CHANGE!!! {}=>{}\n".format(config.model,model))
        config.model=model

    t0 = time.time()
    n_dict = N_Dict(config)
    n_dict.Load("Si3N4", os.path.join(FOLDER_PATH, "Si3N4_310nm-14280nm.txt"), scale=1000);
    n_dict.Load("Graphene", os.path.join(FOLDER_PATH,"Graphene_240nm-30000nm.txt"));
    n_dict.InitMap2(["Si3N4", "Graphene"], config.lenda_tic)
    config.n_dict = n_dict
    return config

class Thin_Layer(object):
    def __init__(self, mater, thick_=np.nan):
        self.material = mater
        self.thick = thick_
        self.n_ = 1 + 0j

class GraSi3N4(Thin_Film_Filters):
    def InitThick(self, thicks=None, h_G=0.35):
        # h_G,hSi=0.35,5        # 计算石墨烯折射率的时候取的厚度是0.35nm
        nLayer = self.config.nLayer
        hSi_1, hSi_2 = self.config.thick_0, self.config.thick_1
        if thicks is None:
            thicks = []
            for no in range(nLayer):
                if no % 2 == 0:
                    thicks.append(h_G)
                else:
                    hSi = random.uniform(hSi_1, hSi_2)
                    thicks.append(hSi)
        self.thicks = thicks
        return self.thicks

    def __init__(self, n_dict, config, thicks=None):
        assert (config is not None)
        self.config = config
        self.n_dict = n_dict
        self.xita = config.xita #angle of incidence
        self.InitThick(thicks)
        self.InitLayers()
        # self.lenda = lenda

        # layer.n_ =
        return

    def InitLayers(self, isAir=True):
        thicks = self.thicks
        nLyaer = len(thicks)
        title = "{} layers, thick=[".format(nLyaer)
        # self.layers.append(Thin_Layer('air'))
        self.layers = []
        no = 0
        for thick in thicks:
            if thick is np.nan:
                mater = 'air'
            else:
                mater = 'Graphene' if no % 2 == 0 else 'Si3N4'
            self.layers.append(Thin_Layer(mater, thick))
            title += "{:.2f},".format(thick)
            no = no + 1
        # self.layers.append(Thin_Layer('air'))
        title += "](nm)"
        self.title = title

    # 基底材料
    def OnSubstrate(self, d0, n0):
        if self.config.model == 'v1':
            d = np.concatenate([np.array([np.nan]), d0, np.array([np.nan])])
            n = np.concatenate([np.array([1.46 + 0j]), n0, np.array([1.0 + 0j])])
        elif self.config.model == 'v0':
            d_air, n_air = np.array([np.nan]), np.array([1 + 0j])
            d = np.concatenate([d_air, d0, d_air])
            n = np.concatenate([n_air, n0, n_air])
        else:
            print( "OnSubstrate:: !!!config.model is {}!!!".format(self.config.model) )
            sys.exit(-66)
        return d, n

    def nMostPt(self):
        # nMost = int((self.config.lenda_1-self.config.lenda_0)/self.config.lenda_step+1)
        nMost = len(self.config.lenda_tic)
        return nMost

    def Chebyshev(self):
        nRow, nCol = self.dataX.shape
        chebX = self.dataX[0, 0:nCol - 1]
        lenda = self.dataX[:, nCol - 1]
        chebY = cheb_fitcurve(lenda, self.dataY[:, 0], 64)
        return

    def CMM(self):
        t0 = time.time()
        nMostRow, nLayer = self.nMostPt(), len(self.layers)
        dataX = np.zeros((nMostRow, nLayer + 1))
        dataY = np.zeros((nMostRow, 3))
        row = 0
        map2 = self.n_dict.map2
        polar = self.config.polar
        for lenda in self.config.lenda_tic:
            #Iterates through each row of x and y-vecs
            i = 0
            # dict_n = {}
            d, n = np.zeros(nLayer), np.zeros(nLayer, dtype=complex)

            for layer in self.layers:
                #Iterates through each col of x
                #This for loop is entirely in charge of generating dataX
                d[i] = layer.thick
                # if layer.material not in dict_n:
                #    dict_n[layer.material] = self.n_dict.Get(layer.material, lenda, isInterPolate=True)
                # n[i] = dict_n[layer.material]   #layer.n_
                if layer.material=='air':
                    n[i] = 1
                else:
                    n[i] = map2[layer.material, lenda]
                dataX[row, i] = d[i]
                i = i + 1
            dataX[row, nLayer] = lenda
            # ^ This piece gets slapped onto dataX at the very end
            # Except for the last piece x is the thickness of each layer... very intuivitve..., it holds whatever each
            # d value holds
            d, n = self.OnSubstrate(d, n)
            #print("d={}\nn={}".format(d, n))

            # y is not built from dataX, it's built from other pieces
            # d is just rows of data after calling OnSubstrate -> n can be calculated by alternatinv each col thru map2
            r, t, R, T, A = jreftran_rt(lenda, d, n, self.xita, polar)
            if False:
                sum = R + T + A
                if abs(sum - 1) > 1.0e-7:
                    print("sum={} R={} T={} A={}".format(sum, R, T, A))
                assert abs(sum - 1) < 1.0e-7
            dataY[row, 0] = R
            dataY[row, 1] = T
            dataY[row, 2] = A

            row = row + 1
            if row >= nMostRow:
                break

        # Sepectrogram(dataY[:, 0])
        # make_melgram(dataY[:, 0],256)
        self.dataX = dataX[0:row, :]
        self.dataY = dataY[0:row, :]

        # print("======N={} time={:.3f}".format(row, time.time() - t0))
        return

def GraSi3N4_sample(nCase, r_seed, sKeyTitle, config, n_dict, x=None):
    np.set_printoptions(linewidth=np.inf)
    random.seed(r_seed - 1)
    np.random.seed(r_seed)
    t0 = time.time()
    arrX, arrY = [], []
    t0 = time.time()
    # 进一步并行     https://stackoverflow.com/questions/9786102/how-do-i-parallelize-a-simple-python-loop
    for case in range(nCase):
        # t1 = time.time()
        filter = GraSi3N4(n_dict, config)
        filter.CMM()
        # filter.Chebyshev()
        # filter.plot_scatter()
        arrX.append(filter.dataX), arrY.append(filter.dataY)
        # gc.collect()
        if case % 100 == 0:
            print("\rno={} time={:.3f} arrX={}\t\t".format(case, time.time() - t0, filter.dataX.shape), end="")

    # https://stackoverflow.com/questions/27516849/how-to-convert-list-of-numpy-arrays-into-single-numpy-array
    mX = np.vstack(arrX)
    mY = np.vstack(arrY)
    nRow = mY.shape[0]
    print("Y[{}] head=\n{} ".format(mY.shape, mY[0:5, :]))
    print("X[{}] head={} ".format(mX.shape, mX[0:5, :]))
    l0, l1, h0, h1 = config.lenda_0, config.lenda_1, config.thick_0, config.thick_1

    out_path = sKeyTitle

    if (nRow < 10000):
        np.savetxt("{}_X{}_.csv".format(out_path, mX.shape), mX, delimiter="\t", fmt='%.8f')
        np.savetxt("{}_Y{}_.csv".format(out_path, mY.shape), mY, delimiter="\t", fmt='%.8f')
    pathZ = "{}_{}_.npz".format(out_path, nRow)
    np.savez_compressed(pathZ, X=mX, Y=mY)
    return pathZ


def X_Curve_Y_thicks(config, mX, mY, nPt, pick_thick=-1):
    pathX = "./data_x.csv"
    pathY = "./data_y.csv"
    # if os.path.isfile(pathZ):

    nCase = (int)(mX.shape[0] / nPt)
    noY = config.noFeat4CMM  # 0,1,2   三条曲线
    n0, n1, pos = 0, 0, pick_thick
    iX = np.zeros((nCase, nPt))
    nLayer = 5 if config.fix_graphene_thick else 10
    iY = np.zeros(nCase) if pick_thick >= 0 else np.zeros((nCase, nLayer))
    x_tic = mX[0:nPt, 10]
    for case in range(nCase):
        n1 = n0 + nPt
        if pos >= 0:
            thick = mX[n0, pos]
            for n in range(n0, n1):
                assert (thick == mX[n0, pos])
            iY[case] = thick
        else:
            iY[case, :] = mX[n0, 1:10:2] if nLayer == 5 else mX[n0, 0:10]
        curve0, curve1 = mY[n0:n1, 0], mY[n0:n1, 1]
        #iX[case, :] = np.concatenate([curve0, curve1])  # mY[n0:n1,noY]
        iX[case, :] = mY[n0:n1,noY]

        n0 = n1
        if False:
            plt.plot(x_tic, iX[case, :])
            plt.show(block=True)

    if config.normal_Y == 1:
        s = (config.thick_1) / 2
        iY = (iY) / s - 1

    np.savetxt(pathX,iY,delimiter=',')
    np.savetxt(pathY,iX,delimiter=',')

    return iY,iX


#@jit
def sind(x):
    y = np.sin(np.radians(x))
    I = x/180.
    if type(I) is not np.ndarray:   #Returns zero for elements where 'X/180' is an integer
        if(I == np.trunc(I) and np.isfinite(I)):
            return 0
    return y

def jreftran_rt(wavelength, d, n, t0, polarization,M=None,M_t=None):
    if M is None:       #在反复多次调用的情况下，可以一次性分配M,M_t
        M = np.zeros((2, 2, d.shape[0]), dtype=complex)
    if M_t is None:
        M_t = np.identity(2, dtype=complex)

    # x = sind(np.array([0,90,180,359, 360]))
    Z0 = 376.730313     #impedance of free space, Ohms
    Y = n / Z0
    g = 1j * 2 * np.pi * n / wavelength        #propagation constant in terms of free space wavelength and refractive index
    t = (n[0] / n * sind(t0))
    t2 = t*t        #python All arithmetic operates elementwise,Array multiplication is not matrix multiplication!!!
    ct = np.sqrt(1 -t2);        # ct=sqrt(1-(n(1)./n*sin(t0)).^2); %cosine theta
    if polarization == 0:       # tilted admittance(斜导纳)
        eta = Y * ct;       # tilted admittance, TE case
    else:
        eta = Y / ct;      # tilted admittance, TM case
    delta = 1j * g * d * ct
    ld = d.shape[0]
    #M = np.zeros((2, 2, ld),dtype=complex)
    for j in range(ld):
        a = delta[j]
        M[0, 0, j] = np.cos(a);
        M[0, 1, j] = 1j / eta[j] * np.sin(a);
        M[1, 0, j] = 1j * eta[j] * np.sin(a);
        M[1, 1, j] = np.cos(a)
        # ("M(:,:,{})={}\n\n".format(j,M[:,:,j]))
    #M_t = np.identity(2,dtype=complex)        #toal charateristic matrix
    for j in range(1,ld - 1):
        M_t = np.matmul(M_t,M[:,:, j])
    # s1 = '({0.real:.2f} + {0.imag:.2f}i)'.format(eta[0])
    # np.set_printoptions(precision=3)
    # print("M_t={}\n\neta={}".format(M_t,eta))

    e_1, e_2 = eta[0], eta[-1]
    #m_1, m_2 = M_t[0, 0] + M_t[0, 1] * e_2, M_t[1, 0] + M_t[1, 1] * e_2
    De = M_t[0, 0] + M_t[0, 1] * eta[-1]
    Nu = M_t[1, 0] + M_t[1, 1] * eta[-1]
    if False:  # Add by Dr.zhu
        Y_tot = (M_t(2, 1) + M_t(2, 2) * eta(len(d))) / (M_t(1, 1) + M_t(1, 2) * eta(len(d)))
        eta_one = eta[0]
        Re = Y_tot.real
        Im = Y_tot.imag
        fx = 2 * Im * eta[0]

    e_de_nu = e_1 * De + Nu
    r = (e_1 * De - Nu) / e_de_nu;
    t = 2 * e_1 / e_de_nu;

    R = abs(r) * abs(r);
    T = (e_2.real / e_1) * abs(t) * abs(t)
    T = T.real
    a = De * np.conj(Nu) - e_2
    A = (4 * e_1 * a.real) / (abs(e_de_nu)*abs(e_de_nu));
    A = A.real
    # return r,t,R,T,A,Y_tot,eta_one,fx,Re,Im
    return r, t, R, T, A

def simulate(Xpred):
    ' Generates y from x data '
    lambda_0 = 240
    lambda_f = 2000
    n_spct = 256

    cheb_plot = cheb.chebpts2(n_spct)
    scale = (lambda_f - lambda_0) / 2
    offset = lambda_0 + scale
    scaled_cheb = [i * scale + offset for i in cheb_plot]

    # Load refractive indices of materials at different sizes
    n_dict = N_Dict()
    n_dict.Load("Si3N4", os.path.join(FOLDER_PATH, "Si3N4_310nm-14280nm.txt"), scale=1000)
    n_dict.Load("Graphene", os.path.join(FOLDER_PATH, "Graphene_240nm-30000nm.txt"))
    n_dict.InitMap2(["Si3N4", "Graphene"], scaled_cheb)
    map2 = n_dict.map2

    Ypred = np.zeros((Xpred.shape[0], n_spct))

    for c, x in enumerate(Xpred):
        dataY = np.zeros((n_spct, 3))

        t_layers = np.array([])
        for val in x:
            # Graphene is preset to always be 0.35 thick
            t_layers = np.concatenate((t_layers, np.array([0.35, val])))
        t_layers = np.concatenate((np.array([np.nan]),t_layers, np.array([np.nan])))

        # Each y-point has different ns in each layer
        for row, lenda in enumerate(scaled_cheb):
            n_layers = np.array([])
            for i in range(len(t_layers) - 2):
                # -2 added to subtract added substrate parameters from length
                if i % 2 == 0:
                    n_layers = np.concatenate((n_layers, np.array([map2['Graphene', lenda]])))
                else:
                    n_layers = np.concatenate((n_layers, np.array([map2['Si3N4', lenda]])))
            # Sandwich by substrate
            n_layers = np.concatenate((np.array([1.46 + 0j]), n_layers, np.array([1 + 0j])))

            # TARGET PARAMETERS OF THE PROBLEM!
            xita = 0
            polar = 0

            r, t, R, T, A = jreftran_rt(lenda, t_layers, n_layers, xita, polar)

            dataY[row, 0] = R
            dataY[row, 1] = T
            dataY[row, 2] = A

        # Because graphs are of absorbance 2nd value is selected
        Ypred[c, :] = dataY[0:n_spct, 2]

    return Ypred


def generate(ndata):
    config = GraSi3N4_init(2018,'v1')
    nLayer = config.nLayer

    sKeyTitle = "_lenda({:.1f}-{:.1f})_H({:.1f}-{:.1f})_N({})_xita({})_polar({})_model({})".format(
        config.lenda_0, config.lenda_1,config.thick_0, config.thick_1,nLayer,config.xita, config.polar, config.model)

    n_dict = N_Dict(config)
    n_dict.Load("Si3N4", "./Si3N4_310nm-14280nm.txt", scale=1000);
    n_dict.Load("Graphene", "./Graphene_240nm-30000nm.txt");
    n_dict.InitMap2(["Si3N4", "Graphene"], config.lenda_tic)

    pathZ = GraSi3N4_sample(ndata, 42, sKeyTitle, config, n_dict)

    with np.load(pathZ) as loaded:
        mX, mY = loaded['X'], loaded['Y']

    nPt = (int)(mX.shape[0] / ndata)
    assert nPt == 256

    x,y = X_Curve_Y_thicks(config, mX, mY, nPt)
    os.remove(pathZ)
    return x,y


if __name__ == '__main__':
    start = time.time()
    isTestLiteMORT = False
    isInverse = True
    ndata = 5000

    generate(ndata)
    print('The generation process of generating {} takes {}s'.format(ndata, time.time() - start))
