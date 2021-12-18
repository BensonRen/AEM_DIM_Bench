"""
The class wrapper for the networks
"""
# Built-in
import os
import time

# Torch
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary
from torch.optim import lr_scheduler
from utils.helper_functions import simulator
# Libs
import numpy as np
from math import inf
import pandas as pd
# Own module
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from utils.time_recorder import time_keeper


class Network(object):
    def __init__(self, model_fn, flags, train_loader, test_loader,
                 ckpt_dir=os.path.join(os.path.abspath(''), 'models'),
                 inference_mode=False, saved_model=None):
        self.model_fn = model_fn                                # The model maker function
        self.flags = flags                                      # The Flags containing the specs
        if inference_mode:                                      # If inference mode, use saved model
            self.ckpt_dir = os.path.join(ckpt_dir, saved_model)
            self.saved_model = saved_model
            print("This is inference mode, the ckpt is", self.ckpt_dir)
        else:                                                   # training mode, create a new ckpt folder
            if flags.model_name is None:
                self.ckpt_dir = os.path.join(ckpt_dir, time.strftime('%Y%m%d_%H%M%S', time.localtime()))
            else:
                self.ckpt_dir = os.path.join(ckpt_dir, flags.model_name)
        self.model = self.create_model()
        # self.encoder, self.decoder, self.spec_enc = self.create_model()     # The model itself
        # self.loss = self.make_loss()                            # The loss function
        self.optm = None                                        # The optimizer: Initialized at train() due to GPU
        self.optm_eval = None                                   # The eval_optimizer: Initialized at eva() due to GPU
        self.lr_scheduler = None                                # The lr scheduler: Initialized at train() due to GPU
        self.train_loader = train_loader                        # The train data loader
        self.test_loader = test_loader                          # The test data loader
        self.log = SummaryWriter(self.ckpt_dir)     # Create a summary writer for keeping the summary to the tensor board
        self.best_validation_loss = float('inf')    # Set the BVL to large number

    def create_model(self):
        """
        Function to create the network module from provided model fn and flags
        :return: the created nn module
        """
        model = self.model_fn(self.flags)
        print(model)
        return model


    def MMD(self, x, y):
        """
        The MDD calculation from https://github.com/VLL-HD/FrEIA/blob/master/experiments/toy_8-modes/toy_8-modes.ipynb
        :param x, y: The samples of 2 distribution we would like to compare
        :return: The Max Mean Discrepency metric on these 2 distributions
        """

        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))

        dxx = rx.t() + rx - 2. * xx
        dyy = ry.t() + ry - 2. * yy
        dxy = rx.t() + ry - 2. * zz

        XX, YY, XY = (torch.zeros(xx.shape).to(device),
                      torch.zeros(xx.shape).to(device),
                      torch.zeros(xx.shape).to(device))

        for a in [0.05, 0.2, 0.9]:
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1

        return torch.mean(XX + YY - 2. * XY)

    def make_loss(self, z):
        """
        Create a tensor that represents the loss. This is consistant both at training time \
        and inference time for Backward model
        :param z: the z output of the model
        :return: the total loss
        """
        ######################################
        # Pytorch nightly modification 02.18 #
        ######################################
        zz = torch.sum(z**2, dim=1)
        jac = self.model.log_jacobian(run_forward=False)                # get the log jacobian
        neg_log_likeli = 0.5 * zz - jac
        return torch.mean(neg_log_likeli), torch.mean(jac), torch.mean(zz)                      # The MSE Loss


    def make_optimizer(self):
        """
        Make the corresponding optimizer from the flags. Only below optimizers are allowed. Welcome to add more
        :return:
        """
        # parameters = [self.encoder.parameters(), self.decoder.parameters(), self.spec_enc.parameters()]
        if self.flags.optim == 'Adam':
            op = torch.optim.Adam(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'RMSprop':
            op = torch.optim.RMSprop(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'SGD':
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
                                              patience=20, verbose=True, threshold=1e-4)

    def save(self):
        """
        Saving the model to the current check point folder with name best_model_forward.pt
        :return: None
        """
        # torch.save(self.model.state_dict, os.path.join(self.ckpt_dir, 'best_model_state_dict.pt'))
        torch.save(self.model, os.path.join(self.ckpt_dir, 'best_model_cINN.pt'))

    def load(self):
        """
        Loading the model from the check point folder with name best_model_forward.pt
        :return:
        """
        # self.model.load_state_dict(torch.load(os.path.join(self.ckpt_dir, 'best_model_state_dict.pt')))
        if torch.cuda.is_available():
            self.model = torch.load(os.path.join(self.ckpt_dir, 'best_model_cINN.pt'))
        else:
            self.model = torch.load(os.path.join(self.ckpt_dir, 'best_model_cINN.pt'), map_location = torch.device('cpu'))


    def train(self):
        """
        The major training function. This would start the training using information given in the flags
        :return: None
        """
        print("Starting training now")
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()

        # Construct optimizer after the model moved to GPU
        self.optm = self.make_optimizer()
        self.lr_scheduler = self.make_lr_scheduler(self.optm)

        dim_x = self.flags.dim_x
        dim_y = self.flags.dim_y
        dim_z = self.flags.dim_z

        # Time keeping
        tk = time_keeper(time_keeping_file=os.path.join(self.ckpt_dir, 'training time.txt'))

        for epoch in range(self.flags.train_step):
            # Set to Training Mode
            train_loss = 0
            self.model.train()
            # If MMD on x-space is present from the start, the model can get stuck.
            # Instead, ramp it up exponetially.
            loss_factor = min(1., 2. * 0.002 ** (1. - (float(epoch) / self.flags.train_step)))

            for j, (x, y) in enumerate(self.train_loader):
                batch_size = len(x)
                # Preparing the data #
                ######################
                if cuda:
                    x = x.cuda()  # Put data onto GPU
                    y = y.cuda()  # Put data onto GPU

                ################
                # Forward step #
                ################
                self.optm.zero_grad()                                   # Zero the gradient first
                z = self.model(x, y)                                    # Get the zpred
                if isinstance(z, tuple):
                    z = z[0]
                #print("type of z is ", type(z))
                #print("length of tuple is", len(z))
                #for content in z:
                #    print("type of content in z: ", type(content))
                #    print(content)
                #    print("shape of content is z:", content.size())
                loss, jac, zz = self.make_loss(z)                                # Make the z loss
                loss.backward()

                ######################
                #  Gradient Clipping #
                ######################
                for parameter in self.model.parameters():
                    parameter.grad.data.clamp_(-self.flags.grad_clamp, self.flags.grad_clamp)

                #########################
                # Descent your gradient #
                #########################
                self.optm.step()                                    # Move one step the optimizer

                train_loss += loss                                 # Aggregate the loss

            # Calculate the avg loss of training
            train_avg_loss = train_loss.cpu().data.numpy() / (j + 1)

            if epoch % self.flags.eval_step == 0:                      # For eval steps, do the evaluations and tensor board
                # Record the training loss to the tensorboard
                self.log.add_scalar('Loss/total_train', train_avg_loss, epoch)
                self.log.add_scalar('Loss/train_jac', jac, epoch)
                self.log.add_scalar('Loss/train_zz', zz, epoch)

                # Set to Evaluation Mode
                self.model.eval()
                print("Doing Evaluation on the model now")

                test_loss = 0
                for j, (x, y) in enumerate(self.test_loader):  # Loop through the eval set
                    batch_size = len(x)

                    ######################
                    # Preparing the data #
                    ######################
                    if cuda:
                        x = x.cuda()  # Put data onto GPU
                        y = y.cuda()  # Put data onto GPU

                    ################
                    # Forward step #
                    ################
                    self.optm.zero_grad()  # Zero the gradient first
                    z = self.model(x, y)  # Get the zpred
                    loss, jac, zz = self.make_loss(z)  # Make the z loss

                    test_loss += loss                                 # Aggregate the loss

                # Record the testing loss to the tensorboard
                test_avg_loss = test_loss.cpu().data.numpy() / (j+1)

                self.log.add_scalar('Loss/total_test', test_avg_loss, epoch)
                self.log.add_scalar('Loss/test_jac', jac, epoch)
                self.log.add_scalar('Loss/test_zz', zz, epoch)

                print("This is Epoch %d, training loss %.5f, validation loss %.5f" \
                      % (epoch, train_avg_loss, test_avg_loss ))

                # Model improving, save the model down
                if test_avg_loss < self.best_validation_loss:
                    self.best_validation_loss = train_avg_loss
                    self.save()
                    print("Saving the model down...")

                    if self.best_validation_loss < self.flags.stop_threshold:
                        print("Training finished EARLIER at epoch %d, reaching loss of %.5f" %\
                              (epoch, self.best_validation_loss))
                        break

            # Learning rate decay upon plateau
            self.lr_scheduler.step(train_avg_loss)
        tk.record(1)                # Record the total time of the training peroid

    def evaluate(self, save_dir='data/', prefix=''):
        self.load()                             # load the model as constructed
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()
        # Set to evaluation mode for batch_norm layers
        self.model.eval()
        # Set the dimensions
        dim_x = self.flags.dim_x
        dim_z = self.flags.dim_z
        saved_model_str = self.saved_model.replace('/', '_') + prefix
        # Get the file names
        Ypred_file = os.path.join(save_dir, 'test_Ypred_{}.csv'.format(saved_model_str))
        Xtruth_file = os.path.join(save_dir, 'test_Xtruth_{}.csv'.format(saved_model_str))
        Ytruth_file = os.path.join(save_dir, 'test_Ytruth_{}.csv'.format(saved_model_str))
        Xpred_file = os.path.join(save_dir, 'test_Xpred_{}.csv'.format(saved_model_str))

        tk = time_keeper(time_keeping_file=os.path.join(save_dir, 'evaluation time.txt'))
        # Open those files to append
        with open(Xtruth_file, 'a') as fxt,open(Ytruth_file, 'a') as fyt,\
                open(Ypred_file, 'a') as fyp, open(Xpred_file, 'a') as fxp:
            # Loop through the eval data and evaluate
            for ind, (x, y) in enumerate(self.test_loader):
                batch_size = len(x)
                # Create a noisy z vector with noise level same as y
                z = torch.randn(batch_size, dim_z, device=device)
                if cuda:
                    x = x.cuda()
                    y = y.cuda()
                Xpred = self.model(z, y, rev=True).cpu().data.numpy()
                np.savetxt(fxt, x.cpu().data.numpy())
                np.savetxt(fxp, Xpred)
                np.savetxt(fyt, y.cpu().data.numpy())
                if self.flags.data_set != 'Yang_sim':
                    Ypred = simulator(self.flags.data_set, Xpred)
                    np.savetxt(fyp, Ypred)
            tk.record(1)
        return Ypred_file, Ytruth_file

    def evaluate_multiple_time(self, time=200, save_dir='/home/sr365/mm_bench_multi_eval/cINN/'):
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
            self.model.cuda()
        self.model.eval()
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
        dim_z = self.flags.dim_z
        batch_size = len(Ytruth_tensor)
        if cuda:
            Ytruth_tensor = Ytruth_tensor.cuda()
        z = torch.randn(batch_size, dim_z, device=device)
        Xpred = self.model(z, Ytruth_tensor, rev=True).cpu().data.numpy()

        # Open those files to append
        with open(Ytruth_file, 'a') as fyt, open(Ypred_file, 'a') as fyp, open(Xpred_file, 'a') as fxp:
            np.savetxt(fyt, Ytruth_tensor.cpu().data.numpy())
            np.savetxt(fxp, Xpred)
            if self.flags.data_set != 'Yang_sim':
                Ypred = simulator(self.flags.data_set, Xpred)
                np.savetxt(fyp, Ypred)
        tk.record(1)
        return Ypred_file, Ytruth_file
