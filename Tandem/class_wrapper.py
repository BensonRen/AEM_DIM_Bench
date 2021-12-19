"""
The class wrapper for the networks
"""
# Built-in
import os
import time
import sys

# Torch
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary
from torch.optim import lr_scheduler

# Libs
import numpy as np
from math import inf
import pandas as pd

# Own module
from utils.time_recorder import time_keeper
from utils.helper_functions import simulator


class Network(object):
    def __init__(self, model_fn_f, model_fn_b, flags, train_loader, test_loader,
                 ckpt_dir=os.path.join(os.path.abspath(''), 'models'),
                 inference_mode=False, saved_model=None):
        """
        The initializer of the Network class which is the wrapper of our neural network, this is for the Tandem model
        :param model_fn_f:
        :param model_fn_b:
        :param flags:
        :param train_loader:
        :param test_loader:
        :param ckpt_dir:
        :param inference_mode:
        :param saved_model:
        """
        self.model_fn_f = model_fn_f                                # The model maker function for forward
        self.model_fn_b = model_fn_b                                # The model maker function for backward
        self.load_forward_ckpt_dir = flags.load_forward_ckpt_dir    # The pre-trained forward model ckpt dir
        print("The pretrian model to load is:", self.load_forward_ckpt_dir)
        self.flags = flags                                      # The Flags containing the specs
        if inference_mode:                                      # If inference mode, use saved model
            self.ckpt_dir = os.path.join(ckpt_dir, saved_model)
            self.saved_model = saved_model
            print("This is inference mode, the ckpt is", self.ckpt_dir)
        else:                                                   # training mode, create a new ckpt folder
            if flags.model_name is None:                    # leave custume name if possible
                self.ckpt_dir = os.path.join(ckpt_dir, time.strftime('%Y%m%d_%H%M%S', time.localtime()))
            else:
                self.ckpt_dir = os.path.join(ckpt_dir, flags.model_name)
        self.model_f, self.model_b = self.create_model()                        # The model itself
        self.loss = self.make_loss()                            # The loss function
        self.optm_f = None                                      # The optimizer: Initialized at train() due to GPU
        self.optm_b = None                                      # The eval_optimizer: Initialized at eva() due to GPU
        self.lr_scheduler = None                                # The lr scheduler: Initialized at train() due to GPU
        self.train_loader = train_loader                        # The train data loader
        self.test_loader = test_loader                          # The test data loader
        self.log = SummaryWriter(self.ckpt_dir)                 # Create a summary writer for keeping the summary to the tensor board
        self.best_validation_loss = float('inf')                # Set the BVL to large number

    def make_optimizer_b(self):
        """
        The function to make the optimizer during evaluation time.
        The difference between optm is that it does not have regularization and it only optmize the self.geometr_eval tensor
        :return: the optimizer_eval
        """
        if self.flags.optim == 'Adam':
            op = torch.optim.Adam(self.model_b.parameters(), lr=self.flags.lr)
        elif self.flags.optim == 'RMSprop':
            op = torch.optim.RMSprop(self.model_b.parameters(), lr=self.flags.lr)
        elif self.flags.optim == 'SGD':
            op = torch.optim.SGD(self.model_b.parameters(), lr=self.flags.lr)
        else:
            raise Exception("Your Optimizer is neither Adam, RMSprop or SGD, please change in param or contact Ben")
        return op

    def create_model(self):
        """
        Function to create the network module from provided model fn and flags
        :return: the created nn module
        """
        model_f = self.model_fn_f(self.flags)
        model_b = self.model_fn_b(self.flags)
        print("Forward model", model_f)
        print("Backward model", model_b)
        return model_f, model_b

    def make_loss(self, logit=None, labels=None, G=None):
        """
        Create a tensor that represents the loss. This is consistant both at training time \
        and inference time for Backward model
        :param logit: The output of the network
        :param labels: The ground truth labels
        :param G: The geometry predicted
        :return: the total loss
        """
        if logit is None:
            return None
        BDY_loss = torch.zeros(size=[], requires_grad=False)
        if torch.cuda.is_available:
            BDY_loss = BDY_loss.cuda()
        # Boundary loss of the geometry_eval to be less than 1
        if G is not None:
            X_range, X_lower_bound, X_upper_bound = self.get_boundary_lower_bound_uper_bound()
            X_mean = (X_lower_bound + X_upper_bound) / 2        # Get the mean
            relu = torch.nn.ReLU()
            BDY_loss_all = 1 * relu(torch.abs(G - self.build_tensor(X_mean)) - 0.5 * self.build_tensor(X_range))
            BDY_loss = torch.mean(BDY_loss_all)
        MSE_loss = nn.functional.mse_loss(logit, labels)          # The MSE Loss
        self.MSE_loss = MSE_loss
        self.Boundary_loss = BDY_loss
        return torch.add(MSE_loss, BDY_loss)

    def build_tensor(self, nparray, requires_grad=False):
        return torch.tensor(nparray, requires_grad=requires_grad, device='cuda', dtype=torch.float)

    def get_boundary_lower_bound_uper_bound(self):
        if self.flags.data_set == 'Chen': 
            dim = 5
        elif self.flags.data_set == 'Peurifoy': 
            dim = 8
        elif self.flags.data_set == 'Yang_sim': 
            dim = 14
        else:
            sys.exit("In Tandem, during getting the boundary loss boundaries, Your data_set entry is not correct, check again!")
        
        return np.array([2 for i in range(dim)]), np.array([-1 for i in range(dim)]), np.array([1 in range(dim)])

    def make_optimizer_f(self):
        """
        Make the corresponding optimizer from the flags. Only below optimizers are allowed. Welcome to add more
        :return:
        """
        if self.flags.optim == 'Adam':
            op = torch.optim.Adam(self.model_f.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'RMSprop':
            op = torch.optim.RMSprop(self.model_f.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'SGD':
            op = torch.optim.SGD(self.model_f.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
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

    def save_f(self):
        """
        Saving the model to the current check point folder with name best_model_forward.pt
        :return: None
        """
        # torch.save(self.model.state_dict, os.path.join(self.ckpt_dir, 'best_model_state_dict.pt'))
        torch.save(self.model_f, os.path.join(self.ckpt_dir, 'best_model_forward.pt'))

    def save_b(self):
        """
        Saving the model to the current check point folder with name best_model_forward.pt
        :return: None
        """
        torch.save(self.model_b, os.path.join(self.ckpt_dir, 'best_model_backward.pt'))

    def load_f(self):
        """
        Only loading the forward model to using the check point folder with name best_model_forward.pt
        :return: None
        """
        if torch.cuda.is_available():           # If load to a CUDA device
            self.model_f = torch.load(os.path.join(self.load_forward_ckpt_dir, 'best_model_forward.pt'))
        else:                                   # If this is a CPU machine
            self.model_f = torch.load(os.path.join(self.load_forward_ckpt_dir, 'best_model_forward.pt'),
                                      map_location=torch.device('cpu'))

    def load(self):
        """
        Loading the model from the check point folder with name best_model_forward.pt
        :return:
        """
        #print("loading backward model, forward can be found in pre-train")
        print("loading from model:", self.ckpt_dir)
        if torch.cuda.is_available():           # If load to a CUDA device
            self.model_f = torch.load(os.path.join(self.ckpt_dir, 'best_model_forward.pt'))
            self.model_b = torch.load(os.path.join(self.ckpt_dir, 'best_model_backward.pt'))
        else:                                   # If this is a CPU machine
            self.model_f = torch.load(os.path.join(self.ckpt_dir, 'best_model_forward.pt'),
                                     map_location=torch.device('cpu'))
            self.model_b = torch.load(os.path.join(self.ckpt_dir, 'best_model_backward.pt'),
                                      map_location=torch.device('cpu'))


    def train(self):
        """
        The major training function. This would start the training using information given in the flags
        :return: None
        """
        """
        Forward Training part
        """
        self.best_forward_validation_loss = self.best_validation_loss

        # Time keeping
        tk = time_keeper(time_keeping_file=os.path.join(self.ckpt_dir, 'training time.txt'))

        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model_f.cuda()
            self.model_b.cuda()
        if self.load_forward_ckpt_dir is None:
            print("Start Forward Training now")
            # Construct optimizer after the model moved to GPU
            self.optm_f = self.make_optimizer_f()
            self.lr_scheduler = self.make_lr_scheduler(self.optm_f)

            for epoch in range(self.flags.train_step):
                # Set to Training Mode
                train_loss = 0
                # boundary_loss = 0                 # Unnecessary during training since we provide geometries
                self.model_f.train()
                for j, (geometry, spectra) in enumerate(self.train_loader):
                    if cuda:
                        geometry = geometry.cuda()                          # Put data onto GPU
                        spectra = spectra.cuda()                            # Put data onto GPU
                    self.optm_f.zero_grad()                                   # Zero the gradient first
                    S_out = self.model_f(geometry)     # Get the output
                    loss = self.make_loss(S_out, spectra)                   # Get the loss tensor
                    loss.backward()                                         # Calculate the backward gradients
                    self.optm_f.step()                                      # Move one step the optimizer
                    train_loss += loss                                      # Aggregate the loss

                # Calculate the avg loss of training
                train_avg_loss = train_loss.cpu().data.numpy() / (j + 1)

                if epoch % self.flags.eval_step == 0:                      # For eval steps, do the evaluations and tensor board
                    # Record the training loss to the tensorboard
                    self.log.add_scalar('Loss/forward_train', train_avg_loss, epoch)
                    # self.log.add_scalar('Loss/BDY_train', boundary_avg_loss, epoch)

                    # Set to Evaluation Mode
                    self.model_f.eval()
                    print("Doing Evaluation on the forward model now")
                    test_loss = 0
                    for j, (geometry, spectra) in enumerate(self.test_loader):  # Loop through the eval set
                        if cuda:
                            geometry = geometry.cuda()
                            spectra = spectra.cuda()
                        logit = self.model_f(geometry)
                        loss = self.make_loss(logit, spectra)                   # compute the loss
                        test_loss += loss                                       # Aggregate the loss

                    # Record the testing loss to the tensorboard
                    test_avg_loss = test_loss.cpu().data.numpy() / (j+1)
                    self.log.add_scalar('Loss/forward_test', test_avg_loss, epoch)

                    print("This is Epoch %d, training loss %.5f, validation loss %.5f" \
                          % (epoch, train_avg_loss, test_avg_loss ))

                    # Model improving, save the model down
                    if test_avg_loss < self.best_forward_validation_loss:
                        self.best_forward_validation_loss = test_avg_loss
                        self.save_f()
                        print("Saving the model down...")

                        if self.best_forward_validation_loss < self.flags.stop_threshold:
                            print("Training finished EARLIER at epoch %d, reaching loss of %.5f" %\
                                  (epoch, self.best_forward_validation_loss))
                            break

                # Learning rate decay upon plateau
                self.lr_scheduler.step(train_avg_loss)
        else:
            print("Loading the pre-trained forward model instead of training it")
            self.load_f()
            # Set to Evaluation Mode
            self.model_f.eval()
            print("Doing Evaluation on the forward model now")
            test_loss = 0
            for j, (geometry, spectra) in enumerate(self.test_loader):  # Loop through the eval set
                if cuda:
                    geometry = geometry.cuda()
                    spectra = spectra.cuda()
                logit = self.model_f(geometry)
                loss = self.make_loss(logit, spectra)  # compute the loss
                test_loss += loss  # Aggregate the loss

            # Record the testing loss to the tensorboard
            test_avg_loss = test_loss.cpu().data.numpy() / (j + 1)
            print("Test loss of forward model loaded is", test_avg_loss)

        """
        Backward Training Part
        """
        self.model_b.train()
        self.model_f.eval()
        print("Now, start Backward Training")
        # Construct optimizer after the model moved to GPU
        self.optm_b = self.make_optimizer_b()
        self.lr_scheduler = self.make_lr_scheduler(self.optm_b)

        for epoch in range(self.flags.train_step):
            # Set to Training Mode
            train_loss = 0
            # boundary_loss = 0                 # Unnecessary during training since we provide geometries
            self.model_b.train()
            self.model_f.eval()
            for j, (geometry, spectra) in enumerate(self.train_loader):
                if cuda:
                    geometry = geometry.cuda()  # Put data onto GPU
                    spectra = spectra.cuda()    # Put data onto GPU
                self.optm_b.zero_grad()         # Zero the gradient first
                G_out = self.model_b(spectra)  # Get the geometry prediction
                S_out = self.model_f(G_out)     # Get the spectra prediction
                loss = self.make_loss(S_out, spectra, G=G_out)  # Get the loss tensor
                loss.backward()  # Calculate the backward gradients
                self.optm_b.step()  # Move one step the optimizer
                train_loss += loss  # Aggregate the loss

            # Calculate the avg loss of training
            train_avg_loss = train_loss.cpu().data.numpy() / (j + 1)
            # boundary_avg_loss = boundary_loss.cpu().data.numpy() / (j + 1)

            if epoch % self.flags.eval_step == 0:  # For eval steps, do the evaluations and tensor board
                # Record the training loss to the tensorboard
                self.log.add_scalar('Loss/backward_train', train_avg_loss, epoch)
                self.log.add_scalar('Loss/BDY_train', self.Boundary_loss.cpu().data.numpy(), epoch)

                # Set to Evaluation Mode
                self.model_b.eval()
                self.model_f.eval()
                print("Doing Evaluation on the backward model now")
                test_loss = 0
                for j, (geometry, spectra) in enumerate(self.test_loader):  # Loop through the eval set
                    if cuda:
                        geometry = geometry.cuda()
                        spectra = spectra.cuda()
                    G_out = self.model_b(spectra)  # Get the geometry prediction
                    S_out = self.model_f(G_out)  # Get the spectra prediction
                    loss = self.make_loss(S_out, spectra, G=G_out)  # compute the loss
                    test_loss += loss  # Aggregate the loss

                # Record the testing loss to the tensorboard
                test_avg_loss = test_loss.cpu().data.numpy() / (j + 1)
                self.log.add_scalar('Loss/backward_test', test_avg_loss, epoch)
                self.log.add_scalar('Loss/BDY_test', self.Boundary_loss.cpu().data.numpy(), epoch)
                
                print("This is Epoch %d, training loss %.5f, validation loss %.5f" \
                      % (epoch, train_avg_loss, test_avg_loss))

                
                # Model improving, save the model down
                if test_avg_loss < self.best_validation_loss:
                    self.best_validation_loss = test_avg_loss
                    self.save_b()
                    print("Saving the backward model down...")

                    if self.best_validation_loss < -1: #self.flags.stop_threshold:
                        print("Training finished EARLIER at epoch %d, reaching loss of %.5f" % \
                              (epoch, self.best_validation_loss))
                        self.log.close()
                        break

            # Learning rate decay upon plateau
            self.lr_scheduler.step(train_avg_loss)
        self.log.close()
        tk.record(1)                # Record the total time of the training peroid
    
    
    def evaluate(self, save_dir='data/', prefix=''):
        self.load()                             # load the model as constructed
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model_b.cuda()
            self.model_f.cuda()

        # Set to evaluation mode for batch_norm layers
        self.model_f.eval()
        self.model_b.eval()


        print('using data set simulator: ',self.flags.data_set)

        saved_model_str = self.saved_model.replace('/','_') + prefix
        # Get the file names
        Ypred_file = os.path.join(save_dir, 'test_Ypred_{}.csv'.format(saved_model_str))
        Xtruth_file = os.path.join(save_dir, 'test_Xtruth_{}.csv'.format(saved_model_str))
        Ytruth_file = os.path.join(save_dir, 'test_Ytruth_{}.csv'.format(saved_model_str))
        Xpred_file = os.path.join(save_dir, 'test_Xpred_{}.csv'.format(saved_model_str))

        tk = time_keeper(os.path.join(save_dir, 'evaluation_time.txt'))
        # Open those files to append
        with open(Xtruth_file, 'a') as fxt,open(Ytruth_file, 'a') as fyt,\
                open(Ypred_file, 'a') as fyp, open(Xpred_file, 'a') as fxp: #,\
            # Loop through the eval data and evaluate
            for ind, (geometry, spectra) in enumerate(self.test_loader):
                if cuda:
                    geometry = geometry.cuda()
                    spectra = spectra.cuda()
                Xpred = self.model_b(spectra)
                Ypred = self.model_f(Xpred).cpu().data.numpy()
                np.savetxt(fyt, spectra.cpu().data.numpy())
                np.savetxt(fxt, geometry.cpu().data.numpy())
                # if self.flags.data_set != 'Yang_sim':
                #     Ypred = simulator(self.flags.data_set, Xpred.cpu().data.numpy())
                #     np.savetxt(fyp, Ypred)
                np.savetxt(fxp, Xpred.cpu().data.numpy())
        tk.record(1)
        return Ypred_file, Ytruth_file

    def evaluate_multiple_time(self, time=200, save_dir='../mm_bench_multi_eval/Tandem/'):
        """
        Make evaluation multiple time for deeper comparison for stochastic algorithms
        :param save_dir: The directory to save the result
        :return:
        """
        save_dir = os.path.join(save_dir, self.flags.data_set)
        tk = time_keeper(os.path.join(save_dir, 'evaluation_time.txt'))
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        for i in range(time):
            self.evaluate(save_dir=save_dir, prefix='inference' + str(i))
            tk.record(i)

    
    def predict(self, Ytruth_file, save_dir='data/', prefix=''):
        self.load()                             # load the model as constructed
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model_b.cuda()
        self.model_b.eval()
        saved_model_str = self.saved_model.replace('/', '_') + prefix

        Ytruth = pd.read_csv(Ytruth_file, header=None, delimiter=',')     # Read the input
        if len(Ytruth.columns) == 1: # The file is not delimitered by ',' but ' '
            Ytruth = pd.read_csv(Ytruth_file, header=None, delimiter=' ')
        Ytruth_tensor = torch.from_numpy(Ytruth.values).to(torch.float)
        print('shape of Ytruth tensor :', Ytruth_tensor.shape)

        # Get the file names
        Ypred_file = os.path.join(save_dir, 'test_Ypred_{}.csv'.format(saved_model_str))
        Ytruth_file = os.path.join(save_dir, 'test_Ytruth_{}.csv'.format(saved_model_str))
        Xpred_file = os.path.join(save_dir, 'test_Xpred_{}.csv'.format(saved_model_str))
        # keep time
        tk = time_keeper(os.path.join(save_dir, 'evaluation_time.txt'))
    
        if cuda:
            Ytruth_tensor = Ytruth_tensor.cuda()
        print('model in eval:', self.model_b)
        Xpred = self.model_b(Ytruth_tensor).detach().cpu().numpy()

        # Open those files to append
        with open(Ytruth_file, 'a') as fyt, open(Ypred_file, 'a') as fyp, open(Xpred_file, 'a') as fxp:
            np.savetxt(fyt, Ytruth_tensor.cpu().data.numpy())
            np.savetxt(fxp, Xpred)
            if self.flags.data_set != 'Yang_sim':
                Ypred = simulator(self.flags.data_set, Xpred)
                np.savetxt(fyp, Ypred)
        tk.record(1)
        return Ypred_file, Ytruth_file
        
