"""
The class wrapper for the networks
"""
# Built-in
import os
import time
import sys
sys.path.append('../utils/')

# Torch
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary
from torch.optim import lr_scheduler

# Libs
import numpy as np
from math import inf
import matplotlib.pyplot as plt
import pandas as pd
# Own module
from utils.time_recorder import time_keeper
from utils.helper_functions import simulator

class Network(object):
    def __init__(self, model_fn, flags, train_loader, test_loader,
                 ckpt_dir=os.path.join(os.path.abspath(''), 'models'),
                 inference_mode=False, saved_model=None):
        self.model_fn = model_fn                                # The model maker function
        self.flags = flags                                      # The Flags containing the specs
        if inference_mode:                                      # If inference mode, use saved model
            if saved_model.startswith('models/'):
                saved_model = saved_model.replace('models/','')
            print(self.ckpt_dir,'\n',saved_model)
            self.ckpt_dir = os.path.join(ckpt_dir, saved_model)
            self.saved_model = saved_model
            print("This is inference mode, the ckpt is", self.ckpt_dir)
        else:                                                   # training mode, create a new ckpt folder
            if flags.model_name is None:                    # leave custume name if possible
                self.ckpt_dir = os.path.join(ckpt_dir, time.strftime('%Y%m%d_%H%M%S', time.localtime()))
            else:
                self.ckpt_dir = os.path.join(ckpt_dir, flags.model_name)
        self.model = self.create_model()                        # The model itself
        self.loss = self.make_loss()                            # The loss function
        self.optm = None                                        # The optimizer: Initialized at train() due to GPU
        self.optm_eval = None                                   # The eval_optimizer: Initialized at eva() due to GPU
        self.lr_scheduler = None                                # The lr scheduler: Initialized at train() due to GPU
        self.train_loader = train_loader                        # The train data loader
        self.test_loader = test_loader                          # The test data loader
        #self.log = SummaryWriter(self.ckpt_dir)                 # Create a summary writer for keeping the summary to the tensor board
        if not os.path.isdir(self.ckpt_dir) and not inference_mode:
            os.mkdir(self.ckpt_dir)
        self.best_validation_loss = float('inf')                # Set the BVL to large number

    def make_optimizer_eval(self, geometry_eval, optimizer_type=None):
        """
        The function to make the optimizer during evaluation time.
        The difference between optm is that it does not have regularization and it only optmize the self.geometr_eval tensor
        :return: the optimizer_eval
        """
        if optimizer_type is None:
            optimizer_type = self.flags.optim
        if optimizer_type == 'Adam':
            op = torch.optim.Adam([geometry_eval], lr=self.flags.lr)
        elif optimizer_type == 'RMSprop':
            op = torch.optim.RMSprop([geometry_eval], lr=self.flags.lr)
        elif optimizer_type == 'SGD':
            op = torch.optim.SGD([geometry_eval], lr=self.flags.lr)
        else:
            raise Exception("Your Optimizer is neither Adam, RMSprop or SGD, please change in param or contact Ben")
        return op

    def create_model(self):
        """
        Function to create the network module from provided model fn and flags
        :return: the created nn module
        """
        model = self.model_fn(self.flags)
        # summary(model, input_size=(128, 8))
        print(model)
        return model

    def make_loss(self, logit=None, labels=None, G=None, return_long=False):
        """
        Create a tensor that represents the loss. This is consistant both at training time \
        and inference time for Backward model
        :param logit: The output of the network
        :param labels: The ground truth labels
        :param larger_BDY_penalty: For only filtering experiments, a larger BDY penalty is added
        :param return_long: The flag to return a long list of loss in stead of a single loss value,
                            This is for the forward filtering part to consider the loss
        :return: the total loss
        """
        if logit is None:
            return None
        MSE_loss = nn.functional.mse_loss(logit, labels)          # The MSE Loss
        BDY_loss = 0
        if G is not None:         # This is using the boundary loss
            X_range, X_lower_bound, X_upper_bound = self.get_boundary_lower_bound_uper_bound()
            X_mean = (X_lower_bound + X_upper_bound) / 2        # Get the mean
            relu = torch.nn.ReLU()
            BDY_loss_all = 1 * relu(torch.abs(G - self.build_tensor(X_mean)) - 0.5 * self.build_tensor(X_range))
            # Meta-material has a higher penalty
            if self.flags.data_set == 'ballistics':
                BDY_loss = 0.001*torch.sum(BDY_loss_all)
            elif self.flags.data_set == 'sine_wave':
                BDY_loss = 0.1*torch.sum(BDY_loss_all)
            elif self.flags.data_set == 'robotic_arm':
                BDY_loss = 0.01*torch.mean(BDY_loss_all)
            else:
                BDY_loss = 10*torch.mean(BDY_loss_all)
            #BDY_loss = self.flags.BDY_strength*torch.sum(BDY_loss_all)
        self.MSE_loss = MSE_loss
        self.Boundary_loss = BDY_loss
        return torch.add(MSE_loss, BDY_loss)


    def build_tensor(self, nparray, requires_grad=False):
        return torch.tensor(nparray, requires_grad=requires_grad, device='cuda', dtype=torch.float)


    def make_optimizer(self, optimizer_type=None):
        """
        Make the corresponding optimizer from the flags. Only below optimizers are allowed. Welcome to add more
        :return:
        """
        # For eval mode to change to other optimizers
        if  optimizer_type is None:
            optimizer_type = self.flags.optim
        if optimizer_type == 'Adam':
            op = torch.optim.Adam(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif optimizer_type == 'RMSprop':
            op = torch.optim.RMSprop(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif optimizer_type == 'SGD':
            op = torch.optim.SGD(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        else:
            raise Exception("Your Optimizer is neither Adam, RMSprop or SGD, please change in param or contact Ben")
        return op

    def make_lr_scheduler(self, optm):
        """
        Make the learning rate scheduler as instructed. More modes can be added to this, current supported ones:
        1. ReduceLROnPlateau (decrease lr when validation error stops improving
        :return:
        """
        return lr_scheduler.ReduceLROnPlateau(optimizer=optm, mode='min',
                                              factor=self.flags.lr_decay_rate,
                                              patience=10, verbose=True, threshold=1e-4)

    def save(self):
        """
        Saving the model to the current check point folder with name best_model_forward.pt
        :return: None
        """
        #torch.save(self.model, os.path.join(self.ckpt_dir, 'best_model_forward.pt'))
        torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir, 'best_model.pt'))

    def load(self):
        """
        Loading the model from the check point folder with name best_model_forward.pt
        :return:
        """
        if torch.cuda.is_available():
            #self.model = torch.load(os.path.join(self.ckpt_dir, 'best_model_forward.pt'))
            self.model.load_state_dict(torch.load(os.path.join(self.ckpt_dir, 'best_model.pt')))
        else:
            #self.model = torch.load(os.path.join(self.ckpt_dir, 'best_model_forward.pt'), map_location=torch.device('cpu'))
            self.model.load_state_dict(torch.load(os.path.join(self.ckpt_dir, 'best_model.pt'), map_location=torch.device('cpu')))

    def train(self):
        """
        The major training function. This would start the training using information given in the flags
        :return: None
        """
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()

        # Construct optimizer after the model moved to GPU
        self.optm = self.make_optimizer()
        self.lr_scheduler = self.make_lr_scheduler(self.optm)

        # Time keeping
        tk = time_keeper(time_keeping_file=os.path.join(self.ckpt_dir, 'training time.txt'))

        for epoch in range(self.flags.train_step):
            # Set to Training Mode
            train_loss = 0
            # boundary_loss = 0                 # Unnecessary during training since we provide geometries
            self.model.train()
            for j, (geometry, spectra) in enumerate(self.train_loader):
                if cuda:
                    geometry = geometry.cuda()                          # Put data onto GPU
                    spectra = spectra.cuda()                            # Put data onto GPU
                self.optm.zero_grad()                               # Zero the gradient first
                logit = self.model(geometry)                        # Get the output
                loss = self.make_loss(logit, spectra)               # Get the loss tensor
                loss.backward()                                     # Calculate the backward gradients
                self.optm.step()                                    # Move one step the optimizer
                train_loss += loss                                  # Aggregate the loss

            # Calculate the avg loss of training
            train_avg_loss = train_loss.cpu().data.numpy() / (j + 1)

            if epoch % self.flags.eval_step == 0:                      # For eval steps, do the evaluations and tensor board
                # Record the training loss to the tensorboard
                #self.log.add_scalar('Loss/train', train_avg_loss, epoch)
                # self.log.add_scalar('Loss/BDY_train', boundary_avg_loss, epoch)

                # Set to Evaluation Mode
                self.model.eval()
                print("Doing Evaluation on the model now")
                test_loss = 0
                for j, (geometry, spectra) in enumerate(self.test_loader):  # Loop through the eval set
                    if cuda:
                        geometry = geometry.cuda()
                        spectra = spectra.cuda()
                    logit = self.model(geometry)
                    loss = self.make_loss(logit, spectra)                   # compute the loss
                    test_loss += loss                                       # Aggregate the loss

                # Record the testing loss to the tensorboard
                test_avg_loss = test_loss.cpu().data.numpy() / (j+1)
                #self.log.add_scalar('Loss/test', test_avg_loss, epoch)

                print("This is Epoch %d, training loss %.5f, validation loss %.5f" \
                      % (epoch, train_avg_loss, test_avg_loss ))

                # Model improving, save the model down
                if test_avg_loss < self.best_validation_loss:
                    self.best_validation_loss = test_avg_loss
                    self.save()
                    print("Saving the model down...")

                    if self.best_validation_loss < self.flags.stop_threshold:
                        print("Training finished EARLIER at epoch %d, reaching loss of %.5f" %\
                              (epoch, self.best_validation_loss))
                        break

            # Learning rate decay upon plateau
            self.lr_scheduler.step(train_avg_loss)
        #self.log.close()
        tk.record(1)                    # Record at the end of the training

    def modulized_bp_ff(self, X_init_mat, Ytruth, FF, save_dir='data/', save_all=True):
        """
        The "evaluation" function for the modulized backprop and forward filtering. It takes the X_init_mat as the different initializations of the X values and do evaluate function on that instead of taking evaluation data from the data loader
        :param X_init_mat: The input initialization of X positions, numpy array of shape (#init, #point, #xdim) usually (2048, 1000, xdim)
        :param Yturth: The Ytruth numpy array of shape (#point, #ydim)
        :param save_dir: The directory to save the results
        :param FF(forward_filtering): The flag to control whether use forward filtering or not
        """
        self.load()                             # load the model as constructed
        try:
            bs = self.flags.backprop_step         # for previous code that did not incorporate this
        except AttributeError:
            print("There is no attribute backprop_step, catched error and adding this now")
            self.flags.backprop_step = 300
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()
        self.model.eval()
        saved_model_str = self.saved_model.replace('/','_')

        # Prepare Ytruth into tensor
        Yt = self.build_tensor(Ytruth, requires_grad=False)
        print("shape of Yt in modulized bp ff is:", Yt.size())
        print("shape of the X_init_mat is:", np.shape(X_init_mat))
        # Loop through #points
        for ind in range(np.shape(X_init_mat)[1]):
            Xpred, Ypred, loss = self.evaluate_one(Yt[ind,:], save_dir=save_dir, save_all=save_all, ind=ind, init_from_Xpred=X_init_mat[:,ind,:], FF=FF)
        return None
            

    def evaluate(self, save_dir='data/', save_all=False, MSE_Simulator=False, save_misc=False, save_Simulator_Ypred=True):
        """
        The function to evaluate how good the Neural Adjoint is and output results
        :param save_dir: The directory to save the results
        :param save_all: Save all the results instead of the best one (T_200 is the top 200 ones)
        :param MSE_Simulator: Use simulator loss to sort (DO NOT ENABLE THIS, THIS IS OK ONLY IF YOUR APPLICATION IS FAST VERIFYING)
        :param save_misc: save all the details that are probably useless
        :param save_Simulator_Ypred: Save the Ypred that the Simulator gives
        (This is useful as it gives us the true Ypred instead of the Ypred that the network "thinks" it gets, which is
        usually inaccurate due to forward model error)
        :return:
        """
        self.load()                             # load the model as constructed
        try:
            bs = self.flags.backprop_step         # for previous code that did not incorporate this
        except AttributeError:
            print("There is no attribute backprop_step, catched error and adding this now")
            self.flags.backprop_step = 300
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()
        self.model.eval()
        saved_model_str = self.saved_model.replace('/','_')
        # Get the file names
        Ypred_file = os.path.join(save_dir, 'test_Ypred_{}.csv'.format(saved_model_str))
        Xtruth_file = os.path.join(save_dir, 'test_Xtruth_{}.csv'.format(saved_model_str))
        Ytruth_file = os.path.join(save_dir, 'test_Ytruth_{}.csv'.format(saved_model_str))
        Xpred_file = os.path.join(save_dir, 'test_Xpred_{}.csv'.format(saved_model_str))
        print("evalution output pattern:", Ypred_file)

        # Time keeping
        tk = time_keeper(time_keeping_file=os.path.join(save_dir, 'evaluation_time.txt'))

        # Open those files to append
        with open(Xtruth_file, 'a') as fxt,open(Ytruth_file, 'a') as fyt,\
                open(Ypred_file, 'a') as fyp, open(Xpred_file, 'a') as fxp:
            # Loop through the eval data and evaluate
            for ind, (geometry, spectra) in enumerate(self.test_loader):
                if cuda:
                    geometry = geometry.cuda()
                    spectra = spectra.cuda()
                # Initialize the geometry first
                Xpred, Ypred, loss = self.evaluate_one(spectra, save_dir=save_dir, save_all=save_all, ind=ind,
                                                        MSE_Simulator=MSE_Simulator, save_misc=save_misc, save_Simulator_Ypred=save_Simulator_Ypred)

                tk.record(ind)                          # Keep the time after each evaluation for backprop
                # self.plot_histogram(loss, ind)                                # Debugging purposes
                np.savetxt(fxt, geometry.cpu().data.numpy())
                np.savetxt(fyt, spectra.cpu().data.numpy())
                if self.flags.data_set != 'Yang':
                    np.savetxt(fyp, Ypred)
                np.savetxt(fxp, Xpred)
        return Ypred_file, Ytruth_file

    def evaluate_one(self, target_spectra, save_dir='data/', MSE_Simulator=False ,save_all=False, ind=None, save_misc=False, save_Simulator_Ypred=True, init_from_Xpred=None, FF=True):
        """
        The function which being called during evaluation and evaluates one target y using # different trails
        :param target_spectra: The target spectra/y to backprop to 
        :param save_dir: The directory to save to when save_all flag is true
        :param MSE_Simulator: Use Simulator Loss to get the best instead of the default NN output logit
        :param save_all: The multi_evaluation where each trail is monitored (instad of the best) during backpropagation
        :param ind: The index of this target_spectra in the batch
        :param save_misc: The flag to print misc information for degbugging purposes, usually printed to best_mse
        :return: Xpred_best: The 1 single best Xpred corresponds to the best Ypred that is being backproped 
        :return: Ypred_best: The 1 singe best Ypred that is reached by backprop
        :return: MSE_list: The list of MSE at the last stage
        :param FF(forward_filtering): [default to be true for historical reason] The flag to control whether use forward filtering or not
        """

        # Initialize the geometry_eval or the initial guess xs
        geometry_eval = self.initialize_geometry_eval(init_from_Xpred)
        # Set up the learning schedule and optimizer
        ######################################################
        # 02.02 for emperically proff of NA bound, SGD optim #
        ######################################################
        self.optm_eval = self.make_optimizer_eval(geometry_eval , optimizer_type='SGD')
        self.lr_scheduler = self.make_lr_scheduler(self.optm_eval)
        
        # expand the target spectra to eval batch size
        target_spectra_expand = target_spectra.expand([self.flags.eval_batch_size, -1])
        
        # Begin NA
        for i in range(self.flags.backprop_step):
            # Make the initialization from [-1, 1], can only be in loop due to gradient calculator constraint
            if init_from_Xpred is None:
                geometry_eval_input = self.initialize_from_uniform_to_dataset_distrib(geometry_eval)
            else:
                geometry_eval_input = geometry_eval
            #if save_misc and ind == 0 and i == 0:                       # save the modified initial guess to verify distribution
            #    np.savetxt('geometry_initialization.csv',geometry_eval_input.cpu().data.numpy())
            self.optm_eval.zero_grad()                                  # Zero the gradient first
            logit = self.model(geometry_eval_input)                     # Get the output
            ###################################################
            # Boundar loss controled here: with Boundary Loss #
            ###################################################
            loss = self.make_loss(logit, target_spectra_expand, G=geometry_eval_input)         # Get the loss
            loss.backward()                                             # Calculate the Gradient
            # update weights and learning rate scheduler
            if i != self.flags.backprop_step - 1:
                self.optm_eval.step()  # Move one step the optimizer
                self.lr_scheduler.step(loss.data)
        
        if save_all:                # If saving all the results together instead of the first one
            ##############################################################
            # Choose the top "trail_nums" points from NA solutions #
            ##############################################################
            # This is not ok because the filtering should take the boundary loss as well!
            # Disabling the sorting!!
            mse_loss = np.reshape(np.sum(np.square(logit.cpu().data.numpy() - target_spectra_expand.cpu().data.numpy()), axis=1), [-1, 1])
            BDY_loss = self.get_boundary_loss_list_np(geometry_eval_input.cpu().data.numpy())
            if self.flags.data_set == 'ballistics':
                BDY_strength = 0.5
            elif self.flags.data_set == 'sine_wave':
                BDY_strength = 0.1
            elif self.flags.data_set == 'robotic_arm':
                BDY_strength = 0.01/2048
            else:
                BDY_strength = 10/2048
            mse_loss += BDY_strength * np.reshape(BDY_loss, [-1, 1])
            # The strategy of re-using the BPed result. Save two versions of file: one with FF and one without
            mse_loss = np.concatenate((mse_loss, np.reshape(np.arange(self.flags.eval_batch_size), [-1, 1])), axis=1)
            ###########################################
            # 02.02 for emperically proff of NA bound #
            ###########################################
            #loss_sort = mse_loss
            loss_sort = mse_loss[mse_loss[:, 0].argsort(kind='mergesort')]                         # Sort the loss list
            loss_sort_FF_off = mse_loss
            exclude_top = 0
            trail_nums = 2048
            good_index = loss_sort[exclude_top:trail_nums+exclude_top, 1].astype('int')                        # Get the indexs
            good_index_FF_off = loss_sort_FF_off[exclude_top:trail_nums+exclude_top, 1].astype('int')                        # Get the indexs
            #print("In save all funciton, the top 10 index is:", good_index[:10])
            if init_from_Xpred is None:
                saved_model_str = self.saved_model.replace('/', '_') + 'inference' + str(ind)
            else:
                saved_model_str = self.saved_model.replace('/', '_') + 'modulized_inference' + str(ind)
            Ypred_file = os.path.join(save_dir, 'test_Ypred_point{}.csv'.format(saved_model_str))
            Yfake_file = os.path.join(save_dir, 'test_Yfake_point{}.csv'.format(saved_model_str))
            Xpred_file = os.path.join(save_dir, 'test_Xpred_point{}.csv'.format(saved_model_str))
            if 'BP_on_FF_on' in save_dir:
                # The strategy of re-using the BPed result. Save two versions of file: one with FF and one without
                save_model_str_FF_off = saved_model_str.replace('BP_on_FF_on', 'BP_on_FF_off')
                Ypred_file_FF_off = Ypred_file.replace('BP_on_FF_on', 'BP_on_FF_off')
                Xpred_file_FF_off = Xpred_file.replace('BP_on_FF_on', 'BP_on_FF_off')
            if self.flags.data_set != 'Yang':  # This is for meta-meterial dataset, since it does not have a simple simulator
                # 2 options: simulator/logit
                Ypred = simulator(self.flags.data_set, geometry_eval_input.cpu().data.numpy())
                if not save_Simulator_Ypred:            # The default is the simulator Ypred output
                    Ypred = logit.cpu().data.numpy()
                if len(np.shape(Ypred)) == 1:           # If this is the ballistics dataset where it only has 1d y'
                    Ypred = np.reshape(Ypred, [-1, 1])
                with open(Xpred_file, 'a') as fxp, open(Ypred_file, 'a') as fyp:
                    np.savetxt(fyp, Ypred[good_index, :])
                    np.savetxt(fxp, geometry_eval_input.cpu().data.numpy()[good_index, :])

                if 'BP_on_FF_on' in save_dir:
                    print("outputting files for FF_off as well")
                    with open(Xpred_file_FF_off, 'a') as fxp, open(Ypred_file_FF_off, 'a') as fyp:
                        np.savetxt(fyp, Ypred[good_index_FF_off, :])
                        np.savetxt(fxp, geometry_eval_input.cpu().data.numpy()[good_index_FF_off, :])
            else:
                with open(Xpred_file, 'a') as fxp:
                    np.savetxt(fxp, geometry_eval_input.cpu().data.numpy()[good_index, :])
                if 'BP_on_FF_on' in save_dir:
                    with open(Xpred_file_FF_off, 'a') as fxp:
                        np.savetxt(fxp, geometry_eval_input.cpu().data.numpy()[good_index_FF_off, :])
            
            ###########################################
            # 02.02 for emperically proff of NA bound #
            ###########################################
            # Save the fake Yp as well
            #with open(Yfake_file, 'a') as fypf:
            #    np.savetxt(fypf, logit.cpu().data.numpy()[good_index, :])

        ###################################
        # From candidates choose the best #
        ###################################
        Ypred = logit.cpu().data.numpy()

        if len(np.shape(Ypred)) == 1:           # If this is the ballistics dataset where it only has 1d y'
            Ypred = np.reshape(Ypred, [-1, 1])
        
        # calculate the MSE list and get the best one
        MSE_list = np.mean(np.square(Ypred - target_spectra_expand.cpu().data.numpy()), axis=1)
        BDY_list = self.get_boundary_loss_list_np(geometry_eval_input.cpu().data.numpy())
        MSE_list += BDY_list
        best_estimate_index = np.argmin(MSE_list)
        #print("The best performing one is:", best_estimate_index)
        Xpred_best = np.reshape(np.copy(geometry_eval_input.cpu().data.numpy()[best_estimate_index, :]), [1, -1])
        if save_Simulator_Ypred and self.flags.data_set != 'Yang':
            Ypred = simulator(self.flags.data_set, geometry_eval_input.cpu().data.numpy())
            if len(np.shape(Ypred)) == 1:           # If this is the ballistics dataset where it only has 1d y'
                Ypred = np.reshape(Ypred, [-1, 1])
        Ypred_best = np.reshape(np.copy(Ypred[best_estimate_index, :]), [1, -1])

        return Xpred_best, Ypred_best, MSE_list


    def get_boundary_loss_list_np(self, Xpred):
        """
        Return the boundary loss in the form of numpy array
        :param Xpred: input numpy array of prediction
        """
        X_range, X_lower_bound, X_upper_bound = self.get_boundary_lower_bound_uper_bound()
        X_mean = (X_lower_bound + X_upper_bound) / 2        # Get the mean
        BDY_loss = np.mean(np.maximum(0, np.abs(Xpred - X_mean) - 0.5*X_range), axis=1)
        return BDY_loss
        

    def initialize_geometry_eval(self, init_from_Xpred):
        """
        Initialize the geometry eval according to different dataset. These 2 need different handling
        :param init_from_Xpred: Initiallize from Xpred file, this is for modulized trails
        :return: The initialized geometry eval

        """
        if init_from_Xpred is not None:
            geometry_eval = self.build_tensor(init_from_Xpred, requires_grad=True)
        elif self.flags.data_set == 'ballistics':
            bs = self.flags.eval_batch_size
            numpy_geometry = np.zeros([bs, self.flags.linear[0]])
            numpy_geometry[:, 0] = np.random.normal(0, 0.5, size=[bs,])
            numpy_geometry[:, 1] = np.max(np.random.normal(1.5, 0.5, size=[bs,]), 0)
            numpy_geometry[:, 2] = np.radians(np.random.uniform(9, 81, size=[bs,]))
            numpy_geometry[:, 3] = np.random.poisson(15, size=[bs,]) / 15
            geometry_eval = self.build_tensor(numpy_geometry, requires_grad=True)
        elif self.flags.data_set == 'robotic_arm':
            bs = self.flags.eval_batch_size
            numpy_geometry = np.random.normal(0, 0.5, size=[bs, 4])
            numpy_geometry[:, 0] /= 2
            geometry_eval = self.build_tensor(numpy_geometry, requires_grad=True)
            print("robotic_arm specific initialization")
        else:
            geometry_eval = torch.rand([self.flags.eval_batch_size, self.flags.linear[0]], requires_grad=True, device='cuda')
        #geomtry_eval = torch.randn([self.flags.eval_batch_size, self.flags.linear[0]], requires_grad=True, device='cuda')
        return geometry_eval

    def initialize_from_uniform_to_dataset_distrib(self, geometry_eval):
        """
        since the initialization of the backprop is uniform from [0,1], this function transforms that distribution
        to suitable prior distribution for each dataset. The numbers are accquired from statistics of min and max
        of the X prior given in the training set and data generation process
        :param geometry_eval: The input uniform distribution from [0,1]
        :return: The transformed initial guess from prior distribution
        """
        X_range, X_lower_bound, X_upper_bound = self.get_boundary_lower_bound_uper_bound()
        geometry_eval_input = geometry_eval * self.build_tensor(X_range) + self.build_tensor(X_lower_bound)
        if self.flags.data_set == 'robotic_arm' or self.flags.data_set == 'ballistics':
            return geometry_eval
        return geometry_eval_input
        #return geometry_eval

    
    def get_boundary_lower_bound_uper_bound(self):
        """
        Due to the fact that the batched dataset is a random subset of the training set, mean and range would fluctuate.
        Therefore we pre-calculate the mean, lower boundary and upper boundary to avoid that fluctuation. Replace the
        mean and bound of your dataset here
        :return:
        """
        if self.flags.data_set == 'sine_wave': 
            return np.array([2, 2]), np.array([-1, -1]), np.array([1, 1])
        elif self.flags.data_set == 'meta_material':
            return np.array([2.272,2.272,2.272,2.272,2,2,2,2]), np.array([-1,-1,-1,-1,-1,-1,-1,-1]), np.array([1.272,1.272,1.272,1.272,1,1,1,1])
        elif self.flags.data_set == 'ballistics':
            return np.array([2, 2, 1.256, 1.1]), np.array([-1, 0.5, 0.157, 0.46]), np.array([1, 2.5, 1.413, 1.56])
        elif self.flags.data_set == 'robotic_arm':
            return np.array([1.2, 2.4, 2.4, 2.4]), np.array([-0.6, -1.2, -1.2, -1.2]), np.array([0.6, 1.2, 1.2, 1.2])
        else:
            sys.exit("In NA, during initialization from uniform to dataset distrib: Your data_set entry is not correct, check again!")


    def predict(self, Xpred_file, no_save=False, load_state_dict=None):
        """
        The prediction function, takes Xpred file and write Ypred file using trained model
        :param Xpred_file: Xpred file by (usually VAE) for meta-material
        :param no_save: do not save the txt file but return the np array
        :param load_state_dict: If None, load model using self.load() (default way), If a dir, load state_dict from that dir
        :return: pred_file, truth_file to compare
        """
        print("entering predict function")
        if load_state_dict is None:
            self.load()         # load the model in the usual way
        else:
            self.model.load_state_dict(torch.load(load_state_dict))
       
        Ypred_file = Xpred_file.replace('Xpred', 'Ypred')
        Ytruth_file = Ypred_file.replace('Ypred', 'Ytruth')
        Xpred = pd.read_csv(Xpred_file, header=None, delimiter=',')     # Read the input
        if len(Xpred.columns) == 1: # The file is not delimitered by ',' but ' '
            Xpred = pd.read_csv(Xpred_file, header=None, delimiter=' ')
        Xpred.info()
        print(Xpred.head())
        print("Xpred shape", np.shape(Xpred.values))
        Xpred_tensor = torch.from_numpy(Xpred.values).to(torch.float)
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()
            Xpred_tensor = Xpred_tensor.cuda()
        # Put into evaluation mode
        self.model.eval()
        Ypred = self.model(Xpred_tensor)
        print(Ypred.cpu().data.numpy())
        if load_state_dict is not None:
            Ypred_file = Ypred_file.replace('Ypred', 'Ypred' + load_state_dict[-7:-4])
        elif self.flags.model_name is not None:
                Ypred_file = Ypred_file.replace('Ypred', 'Ypred' + self.flags.model_name)
        if no_save:                             # If instructed dont save the file and return the array
            return Ypred.cpu().data.numpy(), Ytruth_file
        np.savetxt(Ypred_file, Ypred.cpu().data.numpy())

        return Ypred_file, Ytruth_file

    def plot_histogram(self, loss, ind):
        """
        Plot the loss histogram to see the loss distribution
        """
        f = plt.figure()
        plt.hist(loss, bins=100)
        plt.xlabel('MSE loss')
        plt.ylabel('cnt')
        plt.suptitle('(Avg MSE={:4e})'.format(np.mean(loss)))
        plt.savefig(os.path.join('data','loss{}.png'.format(ind)))
        return None
