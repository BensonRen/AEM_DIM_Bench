import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set()
from utils import helper_functions
from utils.evaluation_helper import compare_truth_pred
from sklearn.neighbors import NearestNeighbors
from pandas.plotting import table
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from matplotlib.lines import Line2D

def InferenceAccuracyExamplePlot(model_name, save_name, title, sample_num=10,  fig_size=(15,5), random_seed=1,
                                 target_region=[0,300 ]):
    """
    The function to plot the Inference accuracy and compare with FFDS algorithm.
    It takes the Ypred and Ytruth file as input and plot the first <sample_num> of spectras.
    It also takes a random of 10 points to at as the target points.
    :param model_name: The model name as the postfix for the Ytruth file
    :param save_name:  The saving name of the figure
    :param title:  The saving title of the figure
    :param sample_num: The number of sample to plot for comparison
    :param fig_size: The size of the figure
    :param random_seed:  The random seed value
    :param target_region:  The region that the targets get
    :return:
    """
    # Get the prediction and truth file first
    Ytruth_file = os.path.join('data','test_Ytruth_{}.csv'.format(model_name))
    Ypred_file = os.path.join('data','test_Ypred_{}.csv'.format(model_name))
    Ytruth = pd.read_csv(Ytruth_file, header=None, delimiter=' ').values
    Ypred = pd.read_csv(Ypred_file, header=None, delimiter=' ').values

    # Draw uniform random distribution for the reference points
    np.random.seed(random_seed)     # To make sure each time we have same target points
    targets = target_region[0] + (target_region[1] - target_region[0]) * np.random.uniform(low=0, high=1, size=10) # Cap the random numbers within 0-299
    targets = targets.astype("int")
    # Make the frequency into real frequency in THz
    fre_low = 0.86
    fre_high = 1.5
    frequency = fre_low + (fre_high - fre_low)/len(Ytruth[0, :]) * np.arange(300)

    for i in range(sample_num):
        # Start the plotting
        f = plt.figure(figsize=fig_size)
        plt.title(title)
        plt.scatter(frequency[targets], Ytruth[i,targets], label='S*')
        plt.plot(frequency, Ytruth[i,:], label='FFDS')
        plt.plot(frequency, Ypred[i,:], label='Candidate')
        plt.legend()
        plt.ylim([0,1])
        plt.xlim([fre_low, fre_high])
        plt.grid()
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Transmittance")
        plt.savefig(os.path.join('data',save_name + str(i) + '.png'))


def RetrieveFeaturePredictionNMse(model_name):
    """
    Retrieve the Feature and Prediciton values and place in a np array
    :param model_name: the name of the model
    return Xtruth, Xpred, Ytruth, Ypred
    """
    # Retrieve the prediction and truth and prediction first
    feature_file = os.path.join('data', 'test_Xtruth_{}.csv'.format(model_name))
    pred_file = os.path.join('data', 'test_Ypred_{}.csv'.format(model_name))
    truth_file = os.path.join('data', 'test_Ytruth_{}.csv'.format(model_name))
    feat_file = os.path.join('data', 'test_Xpred_{}.csv'.format(model_name))

    # Getting the files from file name
    Xtruth = pd.read_csv(feature_file,header=None, delimiter=' ')
    Xpred = pd.read_csv(feat_file,header=None, delimiter=' ')
    Ytruth = pd.read_csv(truth_file,header=None, delimiter=' ')
    Ypred = pd.read_csv(pred_file,header=None, delimiter=' ')
    
    #retrieve mse, mae
    Ymae, Ymse = compare_truth_pred(pred_file, truth_file) #get the maes of y
    
    print(Xtruth.shape)
    return Xtruth.values, Xpred.values, Ytruth.values, Ypred.values, Ymae, Ymse

def ImportColorBarLib():
    """
    Import some libraries that used in a colorbar plot
    """
    import matplotlib.colors as colors
    import matplotlib.cm as cmx
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib as mpl
    print("import sucessful")
    
    return mpl
  
def UniqueMarkers():
    import itertools
    markers = itertools.cycle(( 'x','1','+', '.', '*','D','v','h'))
    return markers
  
def SpectrumComparisonNGeometryComparison(rownum, colnum, Figsize, model_name, boundary = [-1,1,-1,1]):
    """
    Read the Prediction files and plot the spectra comparison plots
    :param SubplotArray: 2x2 array indicating the arrangement of the subplots
    :param Figsize: the size of the figure
    :param Figname: the name of the figures to save
    :param model_name: model name (typically a list of numebr containing date and time)
    """
    mpl = ImportColorBarLib()    #import lib
    
    Xtruth, Xpred, Ytruth, Ypred, Ymae, Ymse =  RetrieveFeaturePredictionNMse(model_name)  #retrieve features
    print("Ymse shape:",Ymse.shape)
    print("Xpred shape:", Xpred.shape)
    print("Xtrth shape:", Xtruth.shape)
    #Plotting the spectrum comaprison
    f = plt.figure(figsize=Figsize)
    fignum = rownum * colnum
    for i in range(fignum):
      ax = plt.subplot(rownum, colnum, i+1)
      plt.ylabel('Transmission rate')
      plt.xlabel('frequency')
      plt.plot(Ytruth[i], label = 'Truth',linestyle = '--')
      plt.plot(Ypred[i], label = 'Prediction',linestyle = '-')
      plt.legend()
      plt.ylim([0,1])
    f.savefig('Spectrum Comparison_{}'.format(model_name))
    
    """
    Plotting the geometry comparsion, there are fignum points in each plot
    each representing a data point with a unique marker
    8 dimension therefore 4 plots, 2x2 arrangement
    
    """
    #for j in range(fignum):
    pointnum = fignum #change #fig to #points in comparison
    
    f = plt.figure(figsize = Figsize)
    ax0 = plt.gca()
    for i in range(4):
      truthmarkers = UniqueMarkers() #Get some unique markers
      predmarkers = UniqueMarkers() #Get some unique markers
      ax = plt.subplot(2, 2, i+1)
      #plt.xlim([29,56]) #setting the heights limit, abandoned because sometime can't see prediciton
      #plt.ylim([41,53]) #setting the radius limits
      for j in range(pointnum):
        #Since the colored scatter only takes 2+ arguments, plot 2 same points to circumvent this problem
        predArr = [[Xpred[j, i], Xpred[j, i]] ,[Xpred[j, i + 4], Xpred[j, i + 4]]]
        predC = [Ymse[j], Ymse[j]]
        truthplot = plt.scatter(Xtruth[j,i],Xtruth[j,i+4],label = 'Xtruth{}'.format(j),
                                marker = next(truthmarkers),c = 'm',s = 40)
        predplot  = plt.scatter(predArr[0],predArr[1],label = 'Xpred{}'.format(j),
                                c =predC ,cmap = 'jet',marker = next(predmarkers), s = 60)
      
      plt.xlabel('h{}'.format(i))
      plt.ylabel('r{}'.format(i))
      rect = mpl.patches.Rectangle((boundary[0],boundary[2]),boundary[1] - boundary[0], boundary[3] - boundary[2],
																		linewidth=1,edgecolor='r',
                                   facecolor='none',linestyle = '--',label = 'data region')
      ax.add_patch(rect)
      plt.autoscale()
      plt.legend(bbox_to_anchor=(0., 1.02, 1., .102),
                 mode="expand",ncol = 6, prop={'size': 5})#, bbox_to_anchor=(1,0.5))
    
    cb_ax = f.add_axes([0.93, 0.1, 0.02, 0.8])
    cbar = f.colorbar(predplot, cax=cb_ax)
    #f.colorbar(predplot)
    f.savefig('Geometry Comparison_{}'.format(model_name))


class HMpoint(object):
    """
    This is a HeatMap point class where each object is a point in the heat map
    properties:
    1. BV_loss: best_validation_loss of this run
    2. feature_1: feature_1 value
    3. feature_2: feature_2 value, none is there is no feature 2
    """
    def __init__(self, bv_loss, f1, f2 = None, f1_name = 'f1', f2_name = 'f2'):
        self.bv_loss = bv_loss
        self.feature_1 = f1
        self.feature_2 = f2
        self.f1_name = f1_name
        self.f2_name = f2_name
        #print(type(f1))
    def to_dict(self):
        return {
            self.f1_name: self.feature_1,
            self.f2_name: self.feature_2,
            self.bv_loss: self.bv_loss
        }


def HeatMapBVL(plot_x_name, plot_y_name, title,  save_name='HeatMap.png', HeatMap_dir = 'HeatMap',
                feature_1_name=None, feature_2_name=None,
                heat_value_name = 'best_validation_loss'):
    """
    Plotting a HeatMap of the Best Validation Loss for a batch of hyperswiping thing
    First, copy those models to a folder called "HeatMap"
    Algorithm: Loop through the directory using os.look and find the parameters.txt files that stores the
    :param HeatMap_dir: The directory where the checkpoint folders containing the parameters.txt files are located
    :param feature_1_name: The name of the first feature that you would like to plot on the feature map
    :param feature_2_name: If you only want to draw the heatmap using 1 single dimension, just leave it as None
    """
    one_dimension_flag = False          #indication flag of whether it is a 1d or 2d plot to plot
    #Check the data integrity 
    if (feature_1_name == None):
        print("Please specify the feature that you want to plot the heatmap");
        return
    if (feature_2_name == None):
        one_dimension_flag = True
        print("You are plotting feature map with only one feature, plotting loss curve instead")

    #Get all the parameters.txt running related data and make HMpoint objects
    HMpoint_list = []
    df_list = []                        #make a list of data frame for further use
    print("going through folder: ", HeatMap_dir)
    for subdir, dirs, files in os.walk(HeatMap_dir):
        for file_name in files:
            #print("passing file-name:", file_name)
            if (file_name == 'parameters.txt'):
                file_path = os.path.join(subdir, file_name) #Get the file relative path from 
                # df = pd.read_csv(file_path, index_col=0)
                flag = helper_functions.load_flags(subdir)
                flag_dict = vars(flag)
                df = pd.DataFrame()
                for k in flag_dict:
                    df[k] = pd.Series(str(flag_dict[k]), index=[0])
                print(df)
                if (one_dimension_flag):
                    df_list.append(df[[heat_value_name, feature_1_name]])
                    HMpoint_list.append(HMpoint(float(df[heat_value_name][0]), eval(str(df[feature_1_name][0])), 
                                                f1_name = feature_1_name))
                else:
                    if feature_2_name == 'linear_unit':                         # If comparing different linear units
                        # linear_unit has always need to be at the feature_2 and either from linear or linear_f,
                        # If you want to test for linear_b for Tandem, make sure you modify manually here
                        try:
                            df['linear_unit'] = eval(df['linear'][0])[1]
                        except:
                            try:
                                df['linear_unit'] = eval(df['linear_b'][0])[1]
                            except:
                                df['linear_unit'] = eval(df['linear_e'][0])[1]
                        #df['best_validation_loss'] = get_bvl(file_path)
                    if feature_2_name == 'kernel_second':                       # If comparing different kernel convs
                        print(df['conv_kernel_size'])
                        print(type(df['conv_kernel_size']))
                        df['kernel_second'] = eval(df['conv_kernel_size'][0])[1]
                        df['kernel_first'] = eval(df['conv_kernel_size'][0])[0]
                    df_list.append(df[[heat_value_name, feature_1_name, feature_2_name]])
                    HMpoint_list.append(HMpoint(float(df[heat_value_name][0]),eval(str(df[feature_1_name][0])),
                                                eval(str(df[feature_2_name][0])), feature_1_name, feature_2_name))
    
    print("df_list =", df_list)
    if len(df_list) == 0:
        print("Your df list is empty, which means you probably mis-spelled the folder name or your folder does not have any parameters.txt?")
    #Concatenate all the dfs into a single aggregate one for 2 dimensional usee
    df_aggregate = pd.concat(df_list, ignore_index = True, sort = False)
    df_aggregate = df_aggregate.astype({heat_value_name: 'float'})

    print("before transformation:", df_aggregate)
    [h, w] = df_aggregate.shape
    print('df_aggregate has shape {}, {}'.format(h, w))
    # making the 2d ones with list to be the lenghth (num_layers)
    for i in range(h):
        for j in range(w):
            print('debugging for nan: ', df_aggregate.iloc[i,j])
            if isinstance(df_aggregate.iloc[i,j], str) and 'nan' not in df_aggregate.iloc[i,j]:
                if isinstance(eval(df_aggregate.iloc[i,j]), list):
                    df_aggregate.iloc[i,j] = len(eval(df_aggregate.iloc[i,j]))

    # If the grid is random (making too sparse a signal), aggregate them
    # The signature of a random grid is the unique value of rows for feature is very large
    if len(np.unique(df_aggregate.values[:, -1])) > 0.8 * h:        # If the number of unique features is more than 80%, this is random
        df_aggregate = df_aggregate.astype('float')
        num_bins = 5                                                # Put all random values into 5 bins
        num_items = int(np.floor(h/num_bins))                            # each bins have num_item numbers inside, last one being more
        feature_1_value_list = df_aggregate.values[:, -1]           # Get the values
        feature_2_value_list = df_aggregate.values[:, -2]
        feature_1_order = np.argsort(feature_1_value_list)          # Get the order
        feature_2_order = np.argsort(feature_2_value_list)
        for i in range(num_bins):
            if i != num_bins - 1:
                df_aggregate.iloc[feature_1_order[i*num_items: (i+1)*num_items], -1] = df_aggregate.iloc[feature_1_order[i*num_items], -1]
                df_aggregate.iloc[feature_2_order[i*num_items: (i+1)*num_items], -2] = df_aggregate.iloc[feature_2_order[i*num_items], -2]
            else:
                df_aggregate.iloc[feature_1_order[i*num_items: ], -1] = df_aggregate.iloc[feature_1_order[i*num_items], -1]
                df_aggregate.iloc[feature_2_order[i*num_items: ], -2] = df_aggregate.iloc[feature_2_order[i*num_items], -2]
    
        # df_aggregate.iloc[:, df.columns != heat_value_name] = df_aggregate.iloc[:, df.columns != heat_value_name].round(decimals=3)  
    
    print('type of last number of df_aggregate is', type(df_aggregate.iloc[-1, -1]))
    
    ########################################################################################################      
    ########################################################################################################
    print("after transoformation:",df_aggregate)
    
    #Change the feature if it is a tuple, change to length of it
    for cnt, point in enumerate(HMpoint_list):
        print("For point {} , it has {} loss, {} for feature 1 and {} for feature 2".format(cnt, 
                                                                point.bv_loss, point.feature_1, point.feature_2))
        assert(isinstance(point.bv_loss, float))        #make sure this is a floating number
        if (isinstance(point.feature_1, tuple)):
            point.feature_1 = len(point.feature_1)
        if (isinstance(point.feature_2, tuple)):
            point.feature_2 = len(point.feature_2)

    
    f = plt.figure()
    #After we get the full list of HMpoint object, we can start drawing 
    if (feature_2_name == None):
        print("plotting 1 dimension HeatMap (which is actually a line)")
        HMpoint_list_sorted = sorted(HMpoint_list, key = lambda x: x.feature_1)
        #Get the 2 lists of plot
        bv_loss_list = []
        feature_1_list = []
        for point in HMpoint_list_sorted:
            bv_loss_list.append(point.bv_loss)
            feature_1_list.append(point.feature_1)
        print("bv_loss_list:", bv_loss_list)
        print("feature_1_list:",feature_1_list)
        #start plotting
        plt.plot(feature_1_list, bv_loss_list,'o-')
    else: #Or this is a 2 dimension HeatMap
        print("plotting 2 dimension HeatMap")
        #point_df = pd.DataFrame.from_records([point.to_dict() for point in HMpoint_list])
        df_aggregate = df_aggregate.round(decimals=3)
        df_aggregate = df_aggregate.reset_index()
        df_aggregate.sort_values(feature_1_name, axis=0, inplace=True)
        df_aggregate.sort_values(feature_2_name, axis=0, inplace=True)
        df_aggregate.sort_values(heat_value_name, axis=0, inplace=True)
        print("before dropping", df_aggregate)
        df_aggregate = df_aggregate.drop_duplicates(subset=[feature_1_name, feature_2_name], keep='first')
        print("after dropping", df_aggregate)
        point_df_pivot = df_aggregate.reset_index().pivot(index=feature_1_name, columns=feature_2_name, values=heat_value_name).astype(float)
        point_df_pivot = point_df_pivot.rename({'5': '05'}, axis=1)
        point_df_pivot = point_df_pivot.reindex(sorted(point_df_pivot.columns), axis=1)
        print("pivot=")
        csvname = HeatMap_dir + 'pivoted.csv'
        point_df_pivot.to_csv(csvname, float_format="%.3g")
        print(point_df_pivot)
        sns.heatmap(point_df_pivot, cmap = "YlGnBu")
    plt.xlabel(plot_y_name)                 # Note that the pivot gives reversing labels
    plt.ylabel(plot_x_name)                 # Note that the pivot gives reversing labels
    plt.title(title)
    plt.savefig(save_name)


def PlotPossibleGeoSpace(figname, Xpred_dir, compare_original = False,calculate_diversity = None):
    """
    Function to plot the possible geometry space for a model evaluation result.
    It reads from Xpred_dir folder and finds the Xpred result insdie and plot that result
    :params figname: The name of the figure to save
    :params Xpred_dir: The directory to look for Xpred file which is the source of plotting
    :output A plot containing 4 subplots showing the 8 geomoetry dimensions
    """
    Xpred = helper_functions.get_Xpred(Xpred_dir)
    
    Xtruth = helper_functions.get_Xtruth(Xpred_dir)

    f = plt.figure()
    ax0 = plt.gca()
    print(np.shape(Xpred))
    if (calculate_diversity == 'MST'):
        diversity_Xpred, diversity_Xtruth = calculate_MST(Xpred, Xtruth)
    elif (calculate_diversity == 'AREA'):
        diversity_Xpred, diversity_Xtruth = calculate_AREA(Xpred, Xtruth)

    for i in range(4):
      ax = plt.subplot(2, 2, i+1)
      ax.scatter(Xpred[:,i], Xpred[:,i + 4],s = 3,label = "Xpred")
      if (compare_original):
          ax.scatter(Xtruth[:,i], Xtruth[:,i+4],s = 3, label = "Xtruth")
      plt.xlabel('h{}'.format(i))
      plt.ylabel('r{}'.format(i))
      plt.xlim(-1,1)
      plt.ylim(-1,1)
      plt.legend()
    if (calculate_diversity != None):
        plt.text(-4, 3.5,'Div_Xpred = {}, Div_Xtruth = {}, under criteria {}'.format(diversity_Xpred, diversity_Xtruth, calculate_diversity), zorder = 1)
    plt.suptitle(figname)
    f.savefig(figname+'.png')

def PlotPairwiseGeometry(figname, Xpred_dir):
    """
    Function to plot the pair-wise scattering plot of the geometery file to show
    the correlation between the geometry that the network learns
    """
    
    Xpredfile = helper_functions.get_Xpred(Xpred_dir)
    Xpred = pd.read_csv(Xpredfile, header=None, delimiter=' ')
    f=plt.figure()
    axes = pd.plotting.scatter_matrix(Xpred, alpha = 0.2)
    #plt.tight_layout()
    plt.title("Pair-wise scattering of Geometery predictions")
    plt.savefig(figname)

def calculate_AREA(Xpred, Xtruth):
    """
    Function to calculate the area for both Xpred and Xtruth under using the segmentation of 0.01
    """
    area_list = np.zeros([2,4])
    X_list = [Xpred, Xtruth]
    binwidth = 0.05
    for cnt, X in enumerate(X_list):
        for i in range(4):
            hist, xedges, yedges = np.histogram2d(X[:,i],X[:,i+4], bins = np.arange(-1,1+binwidth,binwidth))
            area_list[cnt, i] = np.mean(hist > 0)
    X_histgt0 = np.mean(area_list, axis = 1)
    assert len(X_histgt0) == 2
    return X_histgt0[0], X_histgt0[1]

def calculate_MST(Xpred, Xtruth):
    """
    Function to calculate the MST for both Xpred and Xtruth under using the segmentation of 0.01
    """

    MST_list = np.zeros([2,4])
    X_list = [Xpred, Xtruth]
    for cnt, X in enumerate(X_list):
        for i in range(4):
            points = X[:,i:i+5:4]
            distance_matrix_points = distance_matrix(points,points, p = 2)
            csr_mat = csr_matrix(distance_matrix_points)
            Tree = minimum_spanning_tree(csr_mat)
            MST_list[cnt,i] = np.sum(Tree.toarray().astype(float))
    X_MST = np.mean(MST_list, axis = 1)
    return X_MST[0], X_MST[1]


def get_bvl(file_path):
    """
    This is a helper function for 0119 usage where the bvl is not recorded in the pickled object but in .txt file and needs this funciton to retrieve it
    """
    df = pd.read_csv(file_path, delimiter=',')
    bvl = 0
    for col in df:
        if 'best_validation_loss' in col:
            print(col)
            strlist = col.split(':')
            #print("in get_bvl, str is: ", strlist[1])
            if strlist[1].endswith(']') or strlist[1].endswith('}') :
                strlist[1] = strlist[1][:-1]
            bvl = eval(strlist[1])
            print("bvl = ", bvl)
    if bvl == 0:
        print("Error! We did not found a bvl in .txt.file")
    else:
        return float(bvl)



def get_xpred_ytruth_xtruth_from_folder(data_dir):
    """
    This function get the list of Xpred and single Ytruth file in the folder from multi_eval and output the list of Xpred, single Ytruth and Single Xtruth numpy array for further operation
    Since this is not operating on NA series, there is no order in the Xpred files
    ###########################################################
    #NOTE: THIS FUNCTION SHOULD NOT OPERATE ON NA BASED METHOD#
    ###########################################################
    :param data_dir: The directory to get the files
    :output Xpred_list: The list of Xpred files, each element is a numpy array with same shape of Xtruth
    :output Ytruth: The Ytruth numpy array
    :output Xtruth: The Xtruth numpy array
    """
    # Reading Truth files
    Yt = pd.read_csv(os.path.join(data_dir, 'Ytruth.csv'), header=None, delimiter=' ').values
    Xt = pd.read_csv(os.path.join(data_dir, 'Xtruth.csv'), header=None, delimiter=' ').values
    # Reading the list of prediction files
    Xpred_list = []
    for files in os.listdir(data_dir):
        if 'Xpred' in files:
            Xp = pd.read_csv(os.path.join(data_dir, files), header=None, delimiter=' ').values
            Xpred_list.append(Xp)
    return Xpred_list, Xt, Yt

def reshape_xpred_list_to_mat(Xpred_list):
    """
    This function reshapes the Xpred list (typically from "get_xpred_ytruth_xtruth_from_folder") which has the shape: #initialization (2048, as a list) x #data_point (1000) x #xdim 
    into a matrix form for easier formatting for the backpropagation in NA modes
    :param Xpred_list: A list of #init, each element has shape of (#data_point 1000 x #xdim)
    :output X_init_mat: A matrix of shape (2048, 1000, dxim)
    """
    # Get length of list (2048)
    list_length = len(Xpred_list)
    # Get shape of Xpred files
    xshape = np.shape(Xpred_list[0])
    # Init the big matrix
    X_init_mat = np.zeros([list_length, xshape[0], xshape[1]])
    # Fill in the matrix
    for ind, xpred in enumerate(Xpred_list):
        X_init_mat[ind,:,:] = np.copy(xpred)
    return X_init_mat

def get_mse_mat_from_folder(data_dir):
    """
    The function to get the mse matrix from the giant folder that contains all the multi_eval files.
    Due to the data structure difference of NA storing, we would handle NA in different way than other algorithms
    :param data_dir: The directory where the data is in
    """
    Yt = pd.read_csv(os.path.join(data_dir, 'Ytruth.csv'), header=None, delimiter=' ').values
    print("shape of ytruth is", np.shape(Yt))
    # Get all the Ypred into list
    Ypred_list = []
    
    ####################################################################
    # Special handling for NA as it output file structure is different #
    ####################################################################
    if 'NA' in data_dir or 'on' in data_dir or 'GA' in data_dir: 
        l, w = np.shape(Yt)
        print("shape of Yt", l,' ', w)
        num_trails = 200
        #num_trails = 2048
        Ypred_mat = np.zeros([l, num_trails, w])
        check_full = np.zeros(l)                                     # Safety check for completeness
        # A flag labelling whether the Ypred_mat has been updated, 
        # if this is true and first dimension of Yp is still larger, 
        # There is a problem and we would need to exit
        update_Ypred_mat = False                                    
        for files in os.listdir(data_dir):
            if '_Ypred_' in files:
                Yp = pd.read_csv(os.path.join(data_dir, files), header=None, delimiter=' ').values
                if len(np.shape(Yp)) == 1:                          # For ballistic data set where it is a coloumn only
                    Yp = np.reshape(Yp, [-1, 1])
                print("shape of Ypred file is", np.shape(Yp))
                # Truncating to the top num_trails inferences
                if len(Yp) > num_trails:
                    Yp = Yp[:num_trails,:]
                elif len(Yp) < num_trails and update_Ypred_mat is False:        # Only if this is the first time the num_trail is not equal to that
                    num_trails = len(Yp)
                    Ypred_mat = np.zeros([l, num_trails, w])
                    update_Ypred_mat = True
                number_str = files[:-4].split('inference')[-1]
                print(number_str)
                number = int(files[:-4].split('inference')[-1])
                Ypred_mat[number, :, :] = Yp
                check_full[number] = 1
        assert np.sum(check_full) == l, 'Your list is not complete, {} Ypred files out of {} are present'.format(np.sum(check_full), l)
        # Finished fullfilling the Ypred mat, now fill in the Ypred list as before
        for i in range(num_trails):
            Ypred_list.append(Ypred_mat[:, i, :])
    else:
        for files in os.listdir(data_dir):
            if 'Ypred' in files:
                #print(files)
                Yp = pd.read_csv(os.path.join(data_dir, files), header=None, delimiter=' ').values
                if len(np.shape(Yp)) == 1:                          # For ballistic data set where it is a coloumn only
                    Yp = np.reshape(Yp, [-1, 1])
                #print("shape of Ypred file is", np.shape(Yp))
                Ypred_list.append(Yp)
    # Calculate the large MSE matrix
    mse_mat = np.zeros([len(Ypred_list), len(Yt)])
    print("shape of mse_mat is", np.shape(mse_mat))
    
    for ind, yp in enumerate(Ypred_list):
        if np.shape(yp) != np.shape(Yt):
            print("Your Ypred file shape does not match your ytruth, however, we are trying to reshape your ypred file into the Ytruth file shape")
            print("shape of the Yp is", np.shape(yp))
            print("shape of the Yt is", np.shape(Yt))
            yp = np.reshape(yp, np.shape(Yt))
            if ind == 1:
                print(np.shape(yp))
        # For special case yp = -999, it is out of numerical simulator
        print("shape of np :", np.shape(yp))
        print("shape of Yt :", np.shape(Yt))
        mse = np.nanmean(np.square(yp - Yt), axis=1)
        mse_mat[ind, :] = mse
    print("shape of the yp is", np.shape(yp)) 
    print("shape of mse is", np.shape(mse))


    return mse_mat, Ypred_list
    
def MeanAvgnMinMSEvsTry(data_dir):
    """
    Plot the mean average Mean and Min Squared error over Tries
    :param data_dir: The directory where the data is in
    :param title: The title for the plot
    :return:
    """
    # Read Ytruth file
    if not os.path.isdir(data_dir): 
        print("Your data_dir is not a folder in MeanAvgnMinMSEvsTry function")
        print("Your data_dir is:", data_dir)
        return

    # Get the MSE matrix from the giant folder with multi_eval
    mse_mat, Ypred_list = get_mse_mat_from_folder(data_dir)
        
    # Shuffle array and average results
    shuffle_number = 0
    if shuffle_number > 0:
        # Calculate the min and avg from mat
        mse_min_list = np.zeros([len(Ypred_list), shuffle_number])
        mse_avg_list = np.zeros([len(Ypred_list), shuffle_number])
    
        for shuf in range(shuffle_number):
            rng = np.random.default_rng()
            rng.shuffle(mse_mat)
            for i in range(len(Ypred_list)):
                mse_avg_list[i, shuf] = np.mean(mse_mat[:i+1, :])
                mse_min_list[i, shuf] = np.mean(np.min(mse_mat[:i+1, :], axis=0))
        # Average the shuffled result
        mse_avg_list = np.mean(mse_avg_list, axis=1)
        mse_min_list = np.mean(mse_min_list, axis=1)
    else:               # Currently the results are not shuffled as the statistics are enough
        # Calculate the min and avg from mat
        mse_min_list = np.zeros([len(Ypred_list),])
        mse_avg_list = np.zeros([len(Ypred_list),])
        mse_std_list = np.zeros([len(Ypred_list),])
        mse_quan2575_list = np.zeros([2, len(Ypred_list)])
        cut_front = 0
        for i in range(len(Ypred_list)-cut_front):
            mse_avg_list[i] = np.nanmean(mse_mat[cut_front:i+1+cut_front, :])
            mse_min_list[i] = np.nanmean(np.min(mse_mat[cut_front:i+1+cut_front, :], axis=0))
            mse_std_list[i] = np.nanstd(np.min(mse_mat[cut_front:i+1+cut_front, :], axis=0))
            mse_quan2575_list[0, i] = np.nanpercentile(np.min(mse_mat[cut_front:i+1+cut_front, :], axis=0), 25)
            mse_quan2575_list[1, i] = np.nanpercentile(np.min(mse_mat[cut_front:i+1+cut_front, :], axis=0), 75)

    # Save the list down for further analysis
    np.savetxt(os.path.join(data_dir, 'mse_mat.csv'), mse_mat, delimiter=' ')
    np.savetxt(os.path.join(data_dir, 'mse_avg_list.txt'), mse_avg_list, delimiter=' ')
    np.savetxt(os.path.join(data_dir, 'mse_min_list.txt'), mse_min_list, delimiter=' ')
    np.savetxt(os.path.join(data_dir, 'mse_std_list.txt'), mse_std_list, delimiter=' ')
    np.savetxt(os.path.join(data_dir, 'mse_quan2575_list.txt'), mse_quan2575_list, delimiter=' ')

    # Plotting
    f = plt.figure()
    x_axis = np.arange(len(Ypred_list))
    plt.plot(x_axis, mse_avg_list, label='avg')
    plt.plot(x_axis, mse_min_list, label='min')
    plt.legend()
    plt.xlabel('inference number')
    plt.ylabel('mse error')
    plt.savefig(os.path.join(data_dir,'mse_plot vs time'))
    return None


def MeanAvgnMinMSEvsTry_all(data_dir): # Depth=2 now based on current directory structure
    """
    Do the recursive call for all sub_dir under this directory
    :param data_dir: The mother directory that calls
    :return:
    """
    for dirs in os.listdir(data_dir):
        print("entering :", dirs)
        print("this is a folder?:", os.path.isdir(os.path.join(data_dir, dirs)))
        print("this is a file?:", os.path.isfile(os.path.join(data_dir, dirs)))
        #if this is not a folder 
        if not os.path.isdir(os.path.join(data_dir, dirs)):
            print("This is not a folder", dirs)
            continue
        for subdirs in os.listdir(os.path.join(data_dir, dirs)):
            if os.path.isfile(os.path.join(data_dir, dirs, subdirs, 'mse_min_list.txt')):                               # if this has been done
                continue;
            print("enters folder", subdirs)
            MeanAvgnMinMSEvsTry(os.path.join(data_dir, dirs, subdirs))
    return None


def DrawBoxPlots_multi_eval(data_dir, data_name, save_name='Box_plot'):
    """
    The function to draw the statitstics of the data using a Box plot
    :param data_dir: The mother directory to call
    :param data_name: The data set name
    """
    # Predefine name of mse_mat
    mse_mat_name = 'mse_mat.csv'

    #Loop through directories
    mse_mat_dict = {}
    for dirs in os.listdir(data_dir):
        print(dirs)
        if not os.path.isdir(os.path.join(data_dir, dirs)):# or 'NA' in dirs:
            print("skipping due to it is not a directory")
            continue;
        for subdirs in os.listdir((os.path.join(data_dir, dirs))):
            if subdirs == data_name:
                # Read the lists
                mse_mat = pd.read_csv(os.path.join(data_dir, dirs, subdirs, mse_mat_name),
                                           header=None, delimiter=' ').values
                # Put them into dictionary
                mse_mat_dict[dirs] = mse_mat

    # Get the box plot data
    box_plot_data = []
    for key in sorted(mse_mat_dict.keys()):
        data = mse_mat_dict[key][0, :]
        # data = np.mean(mse_mat_dict[key], axis=1)
        box_plot_data.append(data)
        print('{} avg error is : {}'.format(key, np.mean(data)))

    # Start plotting
    f = plt.figure()
    plt.boxplot(box_plot_data, patch_artist=True, labels=sorted(mse_mat_dict.keys()))
    plt.ylabel('mse')
    ax = plt.gca()
    ax.set_yscale('log')
    plt.savefig(os.path.join(data_dir, data_name + save_name + '.png'))
    return None


def DrawAggregateMeanAvgnMSEPlot(data_dir, data_name, save_name='aggregate_plot', 
                                gif_flag=False, plot_points=200, resolution=None, dash_group='nobody',
                                dash_label='', solid_label='',worse_model_mode=False): # Depth=2 now based on current directory structure
    """
    The function to draw the aggregate plot for Mean Average and Min MSEs
    :param data_dir: The mother directory to call
    :param data_name: The data set name
    :param git_flag: Plot are to be make a gif
    :param plot_points: Number of points to be plot
    :param resolution: The resolution of points
    :return:
    """
    # Predefined name of the avg lists
    min_list_name = 'mse_min_list.txt'
    avg_list_name = 'mse_avg_list.txt'
    std_list_name = 'mse_std_list.txt'
    quan2575_list_name = 'mse_quan2575_list.txt'

    # Loop through the directories
    avg_dict, min_dict, std_dict, quan2575_dict = {}, {}, {}, {}
    for dirs in os.listdir(data_dir):
        # Dont include NA for now and check if it is a directory
        print("entering :", dirs)
        print("this is a folder?:", os.path.isdir(os.path.join(data_dir, dirs)))
        print("this is a file?:", os.path.isfile(os.path.join(data_dir, dirs)))
        if not os.path.isdir(os.path.join(data_dir, dirs)):# or dirs == 'NA':# or 'boundary' in dirs::
            print("skipping due to it is not a directory")
            continue;
        for subdirs in os.listdir((os.path.join(data_dir, dirs))):
            if subdirs == data_name:
                # Read the lists
                # mse_avg_list = pd.read_csv(os.path.join(data_dir, dirs, subdirs, avg_list_name),
                #                            header=None, delimiter=' ').values
                mse_min_list = pd.read_csv(os.path.join(data_dir, dirs, subdirs, min_list_name),
                                           header=None, delimiter=' ').values
                # mse_std_list = pd.read_csv(os.path.join(data_dir, dirs, subdirs, std_list_name),
                                        #    header=None, delimiter=' ').values
                mse_quan2575_list = pd.read_csv(os.path.join(data_dir, dirs, subdirs, quan2575_list_name),
                                           header=None, delimiter=' ').values
                print("The quan2575 error range shape is ", np.shape(mse_quan2575_list))
                print("dirs =", dirs)
                print("shape of mse_min_list is:", np.shape(mse_min_list))
                # Put them into dictionary
                # avg_dict[dirs] = mse_avg_list
                min_dict[dirs] = mse_min_list
                # std_dict[dirs] = mse_std_list
                quan2575_dict[dirs] = mse_quan2575_list
    #print("printing the min_dict", min_dict)
       
    def plotDict(dict, name, data_name=None, logy=False, time_in_s_table=None, avg_dict=None, 
                    plot_points=200,  resolution=None, err_dict=None, color_assign=False, dash_group='nobody',
                    dash_label='', solid_label='', plot_xlabel=False, worse_model_mode=False):
        """
        :param name: the name to save the plot
        :param dict: the dictionary to plot
        :param logy: use log y scale
        :param time_in_s_table: a dictionary of dictionary which stores the averaged evaluation time
                in seconds to convert the graph
        :param plot_points: Number of points to be plot
        :param resolution: The resolution of points
        :param err_dict: The error bar dictionary which takes the error bar input
        :param avg_dict: The average dict for plotting the starting point
        :param dash_group: The group of plots to use dash line
        :param dash_label: The legend to write for dash line
        :param solid_label: The legend to write for solid line
        :param plot_xlabel: The True or False flag for plotting the x axis label or not
        :param worse_model_mode: The True or False flag for plotting worse model mode (1X, 10X, 50X, 100X worse model)
        """
        import matplotlib.colors as mcolors
        if worse_model_mode:
            color_dict = {"(1X": "limegreen", "(10X": "blueviolet", "(50X":"cornflowerblue", "(100X": "darkorange"}
        else:
            # manual color setting
            color_dict = {"VAE": "violet","cINN":"chocolate", "INN":"skyblue", "NA": "limegreen",
                        "MDN": "darkorange", "NN":"tab:blue", "GA":"grey", "Tandem":"tab:red"}
            # Automatic color setting
            # color_dict = {}
            # if len(dict.keys()) < 10:
            #     color_pool = mcolors.TABLEAU_COLORS.keys()
            # else:
            #     color_pool = [*list(mcolors.TABLEAU_COLORS.keys()), *list(mcolors.BASE_COLORS), *list(mcolors.CSS4_COLORS.keys())]
            # print('length of color pool ', len(list(color_pool)))
            # for ind, key in enumerate(dict.keys()):
            #     color_dict[key] = list(color_pool)[ind]
        
        print('number of points to be plotted = ', plot_points)
        f = plt.figure(figsize=[6,3])
        ax = plt.gca()
        ax.spines['bottom'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['right'].set_color('black')
        text_pos = 0.01
        # List for legend
        legend_list = []
        print("All the keys=", dict.keys())
        print("All color keys=", color_dict.keys())

        for key in sorted(dict.keys()):
            ######################################################
            # This is for 02.02 getting the T=1, 50, 1000 result #
            ######################################################
            #text = key.replace('_',' ')+"\n" + ': t1={:.2e},t50={:.2e},t1000={:.2e}'.format(dict[key][0][0], dict[key][49][0], dict[key][999][0])
            #print("printing message on the plot now")
            #plt.text(1, text_pos, text, wrap=True)
            #text_pos /= 5
            
            # Linestyle
            if dash_group is not None and dash_group in key:
                linestyle = 'dashed'
            else:
                linestyle = 'solid'
            

            x_axis = np.arange(len(dict[key])).astype('float')
            x_axis += 1
            if time_in_s_table is not None:
                x_axis *= time_in_s_table[data_name][key]
            #print("printing", name)
            
            print('key = ', key)
            #print(dict[key])
            if err_dict is None:
                if color_assign:
                    line_axis, = plt.plot(x_axis[:plot_points:resolution], dict[key][:plot_points:resolution],c=color_dict[key.split('_')[0]],label=key, linestyle=linestyle)
                else:
                    line_axis, = plt.plot(x_axis[:plot_points:resolution], dict[key][:plot_points:resolution],label=key, linestyle=linestyle)
            else:
                # This is the case where we plot the continuous error curve
                if resolution is None:
                    # For mm_bench, this is not needed
                    #label = key.split('_')[0] 
                    label = key
                    if linestyle == 'dashed':
                        label = None
                    #color_key = key.split('_')[0].split(')')[0] # '_' is for separating BP_ox_FF_ox and ')' is for worse model
                    color_key = key
                    #print("color key = ", color_key)
                    line_axis, = plt.plot(x_axis[:plot_points], dict[key][:plot_points], color=color_dict[color_key], linestyle=linestyle, label=label)
                    lower = - err_dict[key][0, :plot_points] + np.ravel(dict[key][:plot_points])
                    higher = err_dict[key][1, :plot_points] + np.ravel(dict[key][:plot_points])
                    plt.fill_between(x_axis[:plot_points], lower, higher, color=color_dict[color_key], alpha=0.1)
                else:
                    if color_assign:
                        line_axis = plt.errorbar(x_axis[:plot_points:resolution], dict[key][:plot_points:resolution],c=color_dict[key.split('_')[0]], yerr=err_dict[key][:, :plot_points:resolution], label=key.replace('_',' '), capsize=5, linestyle=linestyle)#, errorevery=resolution)#,
                    else:
                        line_axis = plt.errorbar(x_axis[:plot_points:resolution], dict[key][:plot_points:resolution], yerr=err_dict[key][:, :plot_points:resolution], label=key.replace('_',' '), capsize=5, linestyle=linestyle)#, errorevery=resolution)#,
                            #dash_capstyle='round')#, uplims=True, lolims=True)
            legend_list.append(line_axis)
        
        if logy:
            ax = plt.gca()
            ax.set_yscale('log')
        # print(legend_list)
        # legend_list.append(Line2D([0], [0], color='k', linestyle='dashed', lw=1, label=dash_label))
        # legend_list.append(Line2D([0], [0], color='k', linestyle='solid', lw=1, label=solid_label))
        #ax.legend(handles=legend_list, loc=1, ncol=2, prop={'size':8})

        if time_in_s_table is not None and plot_xlabel:
            plt.xlabel('inference time (s)')
        elif plot_xlabel:
            plt.xlabel('# of inference made (T)')
        #plt.ylabel('MSE')
        #plt.xlim([1, plot_points])
        plt.xlim([1, min(plot_points, len(dict[key]))])
        ax = plt.gca()
        # plt.legend(prop={'size': 5})
        plt.xscale('log')
        plt.yscale('log')
        plt.xticks([1, 10, 50, 100, 200],['1','10','50','100','200'])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        # ax.tick_params(axis='x', which='minor', bottom=False)
        ax.tick_params(axis='y', which='minor', left=True)
        # ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.grid(True, axis='both',which='major',color='k',alpha=0.1)
        plt.savefig(os.path.join(data_dir, data_name + save_name + name), transparent=True, dpi=300)
        plt.close('all')
        return
        
        #plt.title(data_name.replace('_',' '), fontsize=20)
        ax = plt.gca()
        
        data_index = int(data_name.split(':')[0].split('D')[-1])
        if data_index % 2 == 0: # If this is a even number
            ax.yaxis.tick_right()
        else:
            ax.yaxis.tick_left()
        if data_index < 3:
            ax.xaxis.tick_top()
        #plt.xticks([1, 10, 20, 30, 40, 50, 100, 200])
        plt.savefig(os.path.join(data_dir, data_name + save_name + name), transparent=True, dpi=300)
        plt.close('all')


    ax = plotDict(min_dict,'_minlog_quan2575.png', plot_points=plot_points, logy=True, avg_dict=avg_dict, err_dict=quan2575_dict, data_name=data_name,
            dash_group=dash_group, dash_label=dash_label, solid_label=solid_label, resolution=resolution, worse_model_mode=worse_model_mode)
    #plotDict(min_dict,'_min_quan2575.png', plot_points, resolution, logy=False, avg_dict=avg_dict, err_dict=quan2575_dict)
    #plotDict(min_dict,'_minlog_std.png', plot_points, resolution, logy=True, avg_dict=avg_dict, err_dict=std_dict)
    #plotDict(min_dict,'_min_std.png', plot_points, resolution, logy=False, avg_dict=avg_dict, err_dict=std_dict)

    # if plot gifs
    if not gif_flag:
        return
    else:
        for i in range(2,20,1):
            plotDict(min_dict, str(i), logy=True, plot_points=i)
        for i in range(20,1000,20):
            plotDict(min_dict, str(i), logy=True, plot_points=i)
    
    return ax



def DrawEvaluationTime(data_dir, data_name, save_name='evaluation_time', logy=False, limit=1000):
    """
    This function is to plot the evaluation time behavior of different algorithms on different data sets
    :param data_dir: The mother directory where all the results are put
    :param data_name: The specific dataset to analysis
    :param save_name: The saving name of the plotted figure
    :param logy: take logrithmic at axis y
    :param limit: the limit of x max
    :return:
    """
    eval_time_dict = {}
    for dirs in os.listdir(data_dir):
        print("entering :", dirs)
        print("this is a folder?:", os.path.isdir(os.path.join(data_dir, dirs)))
        print("this is a file?:", os.path.isfile(os.path.join(data_dir, dirs)))
        if not os.path.isdir(os.path.join(data_dir, dirs)):
            print("skipping due to it is not a directory")
            continue;
        for subdirs in os.listdir((os.path.join(data_dir, dirs))):
            if subdirs == data_name:
                # Read the lists
                eval_time = pd.read_csv(os.path.join(data_dir, dirs, subdirs, 'evaluation_time.txt'),
                                           header=None, delimiter=',').values[:, 1]
                # Put them into dictionary
                eval_time_dict[dirs] = eval_time

    # Plotting
    f = plt.figure()
    for key in sorted(eval_time_dict.keys()):
        average_time = eval_time_dict[key][-1] / len(eval_time_dict[key])
        plt.plot(np.arange(len(eval_time_dict[key])), eval_time_dict[key], label=key + 'average_time={0:.2f}s'.format(average_time))
    plt.legend()
    plt.xlabel('#inference trails')
    plt.ylabel('inference time taken (s)')
    plt.title(data_name + 'evaluation_time')
    plt.xlim([0, limit])
    if logy:
        ax = plt.gca()
        ax.set_yscale('log')
        plt.savefig(os.path.join(data_dir, data_name + save_name + 'logy.png'))
    else:
        plt.savefig(os.path.join(data_dir, data_name + save_name + '.png'))

if __name__ == '__main__':
    # NIPS version 
    #MeanAvgnMinMSEvsTry_all('/work/sr365/NA_compare/')
    #datasets = ['ballistics']
    #datasets = ['meta_material', 'robotic_arm','sine_wave','ballistics']
    #lr_list = ['lr1','lr0.5','lr0.05']
    #for lr in lr_list:
    #    MeanAvgnMinMSEvsTry_all('/work/sr365/NA_compare/'+lr)
    #    for dataset in datasets:
    #        DrawAggregateMeanAvgnMSEPlot('/work/sr365/NA_compare/'+lr, dataset)
    #        
    #DrawAggregateMeanAvgnMSEPlot('/work/sr365/NA_compare/', 'ballistics')
        

    work_dir = '../mm_bench_multi_eval/'
    #work_dir = '/home/sr365/MM_Bench/GA/temp-dat'
    datasets = ['Peurifoy']
    #datasets = ['Yang_sim','Chen','Peurifoy']
    MeanAvgnMinMSEvsTry_all(work_dir)
    for dataset in datasets:
        #DrawAggregateMeanAvgnMSEPlot(work_dir, dataset, resolution=5)
        DrawAggregateMeanAvgnMSEPlot(work_dir, dataset)


    # # NIPS version 
    # #work_dir = '/home/sr365/mm_bench_multi_eval'
    # work_dir = '/home/sr365/MDNA_temp/'
    # #work_dir = '/home/sr365/MDNA/Chen/'
    # #lr_list = [10, 1, 0.1, 0.01, 0.001]
    # #MeanAvgnMinMSEvsTry_all(work_dir)
    # #datasets = ['Yang_sim','Chen','Peurifoy']
    # #datasets = ['Yang_sim']
    # datasets = ['Chen']
    # #datasets = ['Peurifoy']
    # #for lr in lr_list:
    # #for bs in [2048]:
    # for bs in [10, 50, 100, 500, 1000, 2048]:#16384]:
    #     #MeanAvgnMinMSEvsTry_all(work_dir + 'bs_{}'.format(bs))
    #     for dataset in datasets:
    #         #DrawAggregateMeanAvgnMSEPlot(work_dir, dataset, resolution=5)
    #         DrawAggregateMeanAvgnMSEPlot(work_dir + 'bs_{}'.format(bs), dataset)
    #         #DrawAggregateMeanAvgnMSEPlot(work_dir, dataset)

    """
    # NIPS version on Groot
    #work_dir = '/data/users/ben/robotic_stuck/retrain5/'
    work_dir = '/data/users/ben/multi_eval/'
    MeanAvgnMinMSEvsTry_all(work_dir)
    datasets = ['ballistics','robotic_arm']
    ##datasets = ['meta_material', 'robotic_arm','sine_wave','ballistics']
    for dataset in datasets:
        DrawAggregateMeanAvgnMSEPlot(work_dir, dataset)
    """

    # NIPS version for INN robo
    #MeanAvgnMinMSEvsTry_all('/work/sr365/multi_eval/special/')
    #datasets = ['robotic_arm']
    #datasets = ['meta_material', 'robotic_arm','sine_wave','ballistics']
    #for dataset in datasets:
    #    DrawAggregateMeanAvgnMSEPlot('/work/sr365/multi_eval/special', dataset)

    #MeanAvgnMinMSEvsTry_all('/home/sr365/ICML_exp_cINN_ball/')
    #DrawAggregateMeanAvgnMSEPlot('/home/sr365/ICML_exp_cINN_ball', dataset)

    """
    # Modulized version (ICML)
    #data_dir = '/data/users/ben/'  # I am groot!
    data_dir = '/home/sr365/' # quad
    #data_dir = '/work/sr365/'
    algo_list = ['cINN','INN','VAE','MDN','Random'] 
    #algo_list = ''
    for algo in algo_list:
        MeanAvgnMinMSEvsTry_all(data_dir + 'ICML_exp/' + algo + '/')
        #datasets = ['meta_material']
        #datasets = ['robotic_arm','sine_wave','ballistics']
        datasets = ['robotic_arm','sine_wave','ballistics','meta_material']
        for dataset in datasets:
            DrawAggregateMeanAvgnMSEPlot(data_dir+ 'ICML_exp/'+algo+'/', dataset)
    """

    # Modulized version plots (ICML_0120)
    #data_dir = '/data/users/ben/'
    ##data_dir = '/work/sr365/'
    ##algo_list = ['cINN','INN','VAE','MDN','Random'] 
    #algo_list = ['Ball','rob','sine','MM']
    #for algo in algo_list:
    #    #MeanAvgnMinMSEvsTry_all(data_dir + 'ICML_exp_0120/top_ones_' + algo + '/')
    #    #datasets = ['robotic_arm','ballistics']
    #    datasets = ['robotic_arm','sine_wave','ballistics','meta_material']
    #    for dataset in datasets:
    #        DrawAggregateMeanAvgnMSEPlot(data_dir+ 'ICML_exp_0120/top_ones_'+algo+'/', dataset)


    ## Draw the top ones
    #datasets = ['robotic_arm','sine_wave','ballistics','meta_material']
    #draw_dir = '/data/users/ben/best_plot/'
    #for dataset in datasets:
    #    DrawAggregateMeanAvgnMSEPlot(draw_dir + dataset , dataset)


