import os
import time
import torch
import math
from utils.helper_functions import simulator
import numpy as np
from utils.time_recorder import time_keeper
from utils.evaluation_helper import plotMSELossDistrib
from model_maker import Net
import sys
from torch.utils.tensorboard import SummaryWriter

INT_NUM = 1e3   # The integer resolution of the binarization
NUM_BITS = 11
# TODO: Have ga.py only hold GA object, and slip GA_manager functionality into class_wrapper

class GA_manager(object):
    def __init__(self, flags, train_loader, test_loader,
                 ckpt_dir=os.path.join(os.path.abspath(''), 'models'),
                 inference_mode=False, saved_model=None, GA_eval_mode=False):
        self.flags = flags                                      # The Flags containing the specs
        if inference_mode:                                      # If inference mode, use saved model
            if saved_model.startswith('models/'):
                saved_model = saved_model.replace('models/','')
            self.ckpt_dir = os.path.join(ckpt_dir, saved_model)
            self.saved_model = saved_model
            print("This is inference mode, the ckpt is", self.ckpt_dir)
        else:                                                   # training mode, create a new ckpt folder
            if flags.model_name is None:                    # leave custume name if possible
                self.ckpt_dir = os.path.join(ckpt_dir, time.strftime('%Y%m%d_%H%M%S', time.localtime()))
            else:
                self.ckpt_dir = os.path.join(ckpt_dir, flags.model_name)
        self.train_loader = train_loader                        # The train data loader
        self.test_loader = test_loader                          # The test data loader
        if not os.path.isdir(self.ckpt_dir) and not inference_mode:
            os.mkdir(self.ckpt_dir)

        self.model = Net(flags)
        self.load()
        self.model.cuda().eval()
        self.algorithm = GA(flags, self.model, self.make_loss)

        self.best_validation_loss = float('inf')                # Set the BVL to large number

    #TODO: Fill in all necessary class_wrapper functions
    def make_loss(self, logit,labels):
        return torch.mean(torch.pow(logit - labels,2), dim=1)

    def load(self):
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(os.path.join(self.ckpt_dir, 'best_model.pt')))
        else:
            self.model.load_state_dict(torch.load(os.path.join(self.ckpt_dir, 'best_model.pt'), map_location=torch.device('cpu')))

    def evaluate(self, save_dir='data/', save_all=False, MSE_Simulator=False, save_misc=False, save_Simulator_Ypred=False):
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
        try:
            bs = self.flags.generations         # for previous code that did not incorporate this
        except AttributeError:
            print("There is no attribute backprop_step, catched error and adding this now")
            self.flags.generations = 300
        cuda = True if torch.cuda.is_available() else False
        #if cuda:
        #    self.model.cuda()
        #self.model.eval()
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
        with open(Xtruth_file, 'w') as fxt,open(Ytruth_file, 'w') as fyt,\
                open(Ypred_file, 'w') as fyp, open(Xpred_file, 'w') as fxp:

            # Loop through the eval data and evaluate
            for ind, (geometry, spectra) in enumerate(self.test_loader):
                print("SAMPLE: {}".format(ind))

                if cuda:
                    geometry = geometry.cuda()
                    spectra = spectra.cuda()
                # Initialize the geometry first
                Xpred, Ypred, loss = self.evaluate_one(spectra, save_dir=save_dir, save_all=save_all, ind=ind,
                                                        MSE_Simulator=MSE_Simulator, save_misc=save_misc, 
                                                        save_Simulator_Ypred=save_Simulator_Ypred)
                tk.record(ind)                          # Keep the time after each evaluation for backprop
                # self.plot_histogram(loss, ind)                                # Debugging purposes
                continue
                np.savetxt(fxt, geometry.cpu().data.numpy())
                np.savetxt(fyt, spectra.cpu().data.numpy())
                if self.flags.data_set != 'Yang_sim':
                    np.savetxt(fyp, Ypred)
                np.savetxt(fxp, Xpred)

                #if (ind+1)%self.flags.eval_step == 0:
                #    plotMSELossDistrib(Ypred_file,Ytruth_file,self.flags)

                # if ind>(self.flags.xtra-1):
                #     print("THIS BREAK IS HIT!",self.flags.xtra)
                #     break

        return Ypred_file, Ytruth_file

    def evaluate_one(self, target_spectra, save_dir='data/', MSE_Simulator=False, save_all=False, ind=None,
                     save_misc=False, save_Simulator_Ypred=False, init_from_Xpred=None, FF=True):
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

        # expand the target spectra to eval batch size
        target_spectra_expand = target_spectra.expand([self.flags.population, -1])

        self.algorithm.set_pop_and_target(target_spectra_expand)
        # select_gen = [0,5,10,25,50,75,99]
        # store = np.empty((1,len(select_gen)))
        begin = time.time()
        for i in range(self.flags.generations):
            #curr = time.time()
            #print('{}\t'.format(i),end='')
            #print("\r\tGEN: {}\tTIME: {}".format(i,curr-strt),end='')
            logit, loss = self.algorithm.evolve()
            # print('in gen {}, avg_loss = {}'.format(i, np.mean(loss.detach().cpu().numpy())))
        loss = self.make_loss(logit,target_spectra_expand)

        print("Evolution time: {}".format(time.time() - begin))
            # if i in select_gen:
            #     good_index = torch.argmin(loss, dim=0).cpu().data.item()
            #     best_val = loss[good_index].cpu().item()
            #     idx = select_gen.index(i)
            #     store[0][idx] = best_val

        # with open(os.path.join(save_dir, 'best_val_by_gen_and_sample.csv'), 'a') as bfile:
        #     np.savetxt(bfile,store,delimiter=',')

        good_index = torch.argmin(loss,dim=0).cpu().data.numpy()
        geometry_eval_input = self.algorithm.old_gen.cpu().data.numpy()
        return 0, 1, 2
        if save_all:  # If saving all the results together instead of the first one
            mse_loss = np.reshape(np.sum(np.square(logit.cpu().data.numpy() - target_spectra_expand.cpu().data.numpy()), axis=1), [-1, 1])
            # The strategy of re-using the BPed result. Save two versions of file: one with FF and one without
            mse_loss = np.concatenate((mse_loss, np.reshape(np.arange(self.flags.eval_batch_size), [-1, 1])), axis=1)
            loss_sort = mse_loss[mse_loss[:, 0].argsort(kind='mergesort')]                         # Sort the loss list
            exclude_top = 0
            trail_nums = 300
            # print('loss_sort')
            # print(loss_sort)
            good_index = loss_sort[exclude_top:trail_nums+exclude_top, 1].astype('int')                        # Get the indexs
            # print('good index')
            # print(good_index)
            saved_model_str = self.saved_model.replace('/', '_')
            Ypred_file = os.path.join(save_dir, 'test_Ypred_point{}{}{}.csv'.format(saved_model_str,'inference',ind))
            Xpred_file = os.path.join(save_dir, 'test_Xpred_point{}{}{}.csv'.format(saved_model_str,'inference',ind))
            print("HERE:\t",Ypred_file)
            if self.flags.data_set != 'Yang_sim':  # This is for meta-meterial dataset, since it does not have a simple simulator
                Ypred = simulator(self.flags.data_set, geometry_eval_input[good_index, :])
                with open(Xpred_file, 'a') as fxp, open(Ypred_file, 'a') as fyp:
                    np.savetxt(fyp, Ypred)
                    np.savetxt(fxp, geometry_eval_input[good_index, :])
            else:
                with open(Xpred_file, 'a') as fxp:
                    np.savetxt(fxp, geometry_eval_input[good_index, :])


        ###################################
        # From candidates choose the best #
        ###################################
        Ypred = logit.cpu().data.numpy()

        # calculate the MSE list and get the best one
        MSE_list = np.mean(np.square(Ypred - target_spectra_expand.cpu().data.numpy()), axis=1)
        best_estimate_index = np.argmin(MSE_list)
        # print("The best performing one is:", best_estimate_index)
        Xpred_best = np.reshape(np.copy(geometry_eval_input[best_estimate_index, :]), [1, -1])
        if save_Simulator_Ypred and self.flags.data_set != 'Yang_sim':
            begin = time.time()
            Ypred = simulator(self.flags.data_set, geometry_eval_input)
            print("Simulation: ", time.time()-begin)
            if len(np.shape(Ypred)) == 1:  # If this is the ballistics dataset where it only has 1d y'
                Ypred = np.reshape(Ypred, [-1, 1])
        Ypred_best = np.reshape(np.copy(Ypred[best_estimate_index, :]), [1, -1])

        return Xpred_best, Ypred_best, MSE_list



class GA(object):
    def __init__(self,flags,model,loss_fn):
        ' Initialize general GA parameters '
        self.n_params = flags.linear[0]
        self.data_set = flags.data_set
        self.n_elite = flags.elitism
        self.k = flags.k
        self.n_pop = flags.population
        self.n_kids = self.n_pop - self.n_elite
        self.n_pairs = int(math.ceil(self.n_kids/2))
        self.mut = flags.mutation
        self.cross_p = flags.crossover
        self.selector = self.make_selector(flags.selection_operator)
        self.X_op = self.make_X_op(flags.cross_operator)
        self.device = torch.device('cpu' if flags.use_cpu_only else 'cuda')
        self.loss_fn = loss_fn
        self.eval = False
        self.span,self.lower,self.upper = self.get_boundary_lower_bound_upper_bound()

        self.model = model
        self.calculate_spectra = self.sim if self.eval else self.model
        self.target = None
        self.old_gen = None
        self.generation = None
        self.fitness = None
        self.p_child = None

    def get_boundary_lower_bound_upper_bound(self):
        """
        Due to the fact that the batched dataset is a random subset of the training set, mean and range would fluctuate.
        Therefore we pre-calculate the mean, lower boundary and upper boundary to avoid that fluctuation. Replace the
        mean and bound of your dataset here
        :return:
        """
        if self.data_set == 'Chen':
            dim = 5
        elif self.data_set == 'Peurifoy':
            dim = 8
        elif self.data_set == 'Yang_sim':
            dim = 14
        else:
            sys.exit(
                "In GA, during getting the boundary loss boundaries, Your data_set entry is not correct, check again!")
        return 2*torch.ones(dim,device=self.device,requires_grad=False), -torch.ones(dim,device=self.device,requires_grad=False),\
               torch.ones(dim, device=self.device, requires_grad=False)

    def initialize_in_range(self, n_samples):
        return torch.rand(n_samples,self.n_params,device=self.device,requires_grad=False)*self.span + self.lower

    def sim(self,X):
        return torch.from_numpy(simulator(self.data_set,X.cpu().data.numpy())).float().cuda()

    def set_pop_and_target(self, trgt):
        ' Initialize parameters that change with new targets: fitness, target, generation '
        self.target = trgt
        self.target.requires_grad = False
        self.generation = self.initialize_in_range(self.n_pop)
        self.p_child = torch.empty(2*self.n_pairs,self.n_params, device=self.device, requires_grad=False)
        self.fitness = torch.empty(self.n_pop, device=self.device, requires_grad=False)

    def make_selector(self, S_type):
        ' Returns selection operator used to select parent mating pairs '
        if S_type == 'roulette':
            return self.roulette
        elif S_type == 'tournament':
            return self.tournament
        elif S_type == 'decimation':
            return self.decimation
        else:
            raise(Exception('Selection Operator improperly configured'))

    def make_X_op(self, X_type):
        ' Returns crossover operator '
        if X_type == 'uniform':
            return self.u_cross
        elif X_type == 'single-point':
            return self.s_cross
        else:
            raise (Exception('Crossover Operator improperly configured'))

    def roulette(self):
        ' Performs roulette-wheel selection from self.generation using self.fitness '
        mock_fit = 1/self.fitness
        total = torch.sum(mock_fit)
        mock_fit = mock_fit/total

        r_wheel = torch.cumsum(mock_fit,0)
        spin_values = torch.rand(2*self.n_pairs,1, device=self.device, requires_grad=False)

        r_wheel = r_wheel.unsqueeze(0).expand(2*self.n_pairs,-1)
        dist = r_wheel - spin_values
        dist[torch.heaviside(-dist,torch.tensor(0.0, device=self.device,requires_grad=False)).bool()] = 1
        idxs = torch.argmin(dist, dim=1)

        self.p_child = self.generation[idxs,:].clone()
        self.p_child = self.p_child.unsqueeze(1).view(self.n_pairs,2,self.n_params)

    def decimation(self):
        ' Performs population decimation selection from self.generation assuming it is sorted by fitness '
        idxs = torch.randint(self.k,(2*self.n_pairs,),device=self.device,requires_grad=False)
        self.p_child = self.generation[idxs].clone()
        self.p_child = self.p_child.unsqueeze(1).view(self.n_pairs,2,self.n_params)

    def tournament(self):
        ' Performs tournament-style selection from self.generation assuming it is sorted by fitness '
        for row in range(2*self.n_pairs):
            competitors = torch.randperm(self.n_pop,device=self.device,requires_grad=False)[:self.k]
            self.p_child[row] = self.generation[torch.min(competitors)]
        self.p_child = self.p_child.clone().unsqueeze(1).view(self.n_pairs,2,self.n_params)

    def u_cross(self):
        ' Performs uniform crossover given pairs tensor arranged sequentially in parent-pairs '
        rand_vec = torch.rand(self.n_pairs, device=self.device, requires_grad=False)
        idcs = (rand_vec < self.cross_p).nonzero().squeeze()
        n_selected = idcs.shape[0]
        site_mask = torch.heaviside(torch.rand(n_selected,self.n_params, device=self.device, requires_grad=False)-0.5,
                                    torch.tensor(1.0, device=self.device, requires_grad=False)).bool()

        parentC = self.p_child.clone()
        for c,i in enumerate(idcs):
            p0 = self.p_child[i][0]
            p0c = parentC[i][0]
            p1 = self.p_child[i][1]
            p1c = parentC[i][1]

            p0[site_mask[c]] = p1c[site_mask[c]]
            p1[site_mask[c]] = p0c[site_mask[c]]

    def s_cross(self):
        ' Performs single-point crossover given pairs tensor arranged sequentially in parent-pairs '
        rand_vec = torch.rand(self.n_pairs, device=self.device, requires_grad=False)    # Format the shape
        idcs = (rand_vec < self.cross_p).nonzero().squeeze()                            # random threshold
        n_selected = idcs.shape[0]                                                      # Get the number of mutations

        #siteX = torch.randint(1, self.n_params, (n_selected,), device=self.device, requires_grad=False)

        # parentC = self.p_child.clone()
        # # print('in single point cross over, p_child shape =', self.p_child.size())
        # for c,i in enumerate(idcs):
        #     p0 = self.p_child[i][0]
        #     p0c = parentC[i][0]
        #     p1 = self.p_child[i][1]
        #     p1c = parentC[i][1]

        #     p0[siteX[c]:] = p1c[siteX[c]:]
        #     p1[siteX[c]:] = p0c[siteX[c]:]

        """
        Ben's implementation using binary operations
        """

        siteX = torch.randint(1, NUM_BITS*self.n_params, (n_selected,), device=self.device, requires_grad=False)
        for index, site in zip(idcs, siteX):
            # check = self.p_child[index].clone().detach().cpu().numpy()
            # print(self.p_child[index][0])
            # print(self.p_child[index][1])
            temp = self.p_child[index][0].clone()       # Clone a temp to swap
            self.p_child[index][0][site:] = self.p_child[index][1][site:] # Swap them
            self.p_child[index][1][site:] = temp[site:]
            # print('after cross over')
            # print(self.p_child[index][0])
            # print(self.p_child[index][1])
            
            # print('check num')
            # print(np.sum(check[0] != self.p_child[index][0].detach().cpu().numpy()))
            # print(np.sum(check[1] != self.p_child[index][1].detach().cpu().numpy()))
            # quit()


            # for j in range(len(self.p_child[0][0])):
            #     # Create a binary map of x bits to the left all 1 and others all 0
            #     bit_right_map = torch.tensor(2**siteX[index_ind] - 1, dtype=torch.int32, requires_grad=False)
            #     # Mask both parents with this
            #     p1r = self.p_child[index][0][j].bitwise_and(bit_right_map)
            #     p2r = self.p_child[index][1][j].bitwise_and(bit_right_map)
            #     p1l = self.p_child[index][0][j].bitwise_and(bit_right_map.bitwise_not())
            #     p2l = self.p_child[index][1][j].bitwise_and(bit_right_map.bitwise_not())

            #     # print('bit_right_map=')
            #     # print(bin(bit_right_map.detach().cpu().numpy()))
            #     # print('p1r')
            #     # print(bin(p1r.detach().cpu().numpy()))
            #     # print('p1l')
            #     # print(bin(p1l.detach().cpu().numpy()))
            #     # print('before switch, parents = ')
            #     # print(self.p_child[index][0][0].detach().cpu().numpy(),self.p_child[index][1][0].detach().cpu().numpy())
            #     # print(bin(self.p_child[index][0][0].detach().cpu().numpy()),bin(self.p_child[index][1][0].detach().cpu().numpy()))

            #     self.p_child[index][0][j] = p1r.bitwise_or(p2l)
            #     self.p_child[index][1][j] = p2r.bitwise_or(p1l)

            #     # print('switching bit of ', siteX[index_ind])
            #     # print('after switch, child = ') 
            #     # print(bin(self.p_child[index][0][0].detach().cpu().numpy()),bin(self.p_child[index][1][0].detach().cpu().numpy()))

            

    def mutate(self):
        ' Performs single-point random mutation given children tensor '
        # self.p_child = self.p_child.view(2*self.n_pairs,self.n_params)          # Format the shape
        # r_vec = torch.rand_like(self.p_child)                                   # Generate random number for thresholding
        # param_prob = self.mut > r_vec                                           # Threshold random number to determine if mutate or not
        # Ashwin's operation for mutation
        # self.p_child[param_prob] = self.initialize_in_range(self.p_child.shape[0])[param_prob]  # Mutate by re-initialize these "mutated" ones
        # Ben's operation for mutation
        # self.p_child[param_prob] += 0.5*torch.randn_like(self.p_child[param_prob])   # Mutate by re-initialize these "mutated" ones
        
        """
        Ben's implementation using binary operations
        """
        # First get a mask with 5% chance of 1 and all other bits 0, INT_NUM=1e7 so it has 24 bits
        # num_bits = NUM_BITS
        #self.p_child = self.p_child.view(1, -1)          # Format the shape
        
        ##################################################################
        # THe new one liner bit flip, p_child = 1 - p_child since binary #
        ##################################################################
        # print('mutation')
        # print(self.p_child[0])
        self.p_child = self.p_child.type(torch.bool)  # Change to bool type
        # check = self.p_child.clone().detach().cpu().numpy()
        indexs = torch.rand(self.p_child.size(), device=self.p_child.device) < self.mut # Get the index of mutating ones
        self.p_child[indexs] = self.p_child[indexs].logical_not()                       # Mutate them
        # print('diff bits ={}'.format(np.sum(self.p_child.detach().cpu().numpy() != check)))
        # print('out of ', len(np.reshape(check, [-1, 1])))
        self.p_child = self.p_child.type(torch.int32)                                   # Get back to int type
        # print(self.p_child[0])

        # for i in range(len(self.p_child)):
        #     bit_positions = np.arange(num_bits)[np.random.random(size=num_bits) < self.mut]
        #     # For each of the cases where we need to mutate, we would do a mutation by shifting and xor
        #     for bit_pos in bit_positions:
        #         # print('mutate bit pos', bit_pos)
        #         # print('original ')
        #         # print(bin(self.p_child[i][0].detach().cpu().numpy()))
        #         bit_operator = torch.tensor(int(1) << bit_pos)
        #         #print(bit_operator)
        #         self.p_child[i] = self.p_child[i].bitwise_xor(bit_operator)
        #         # print('after mutation')
        #         # print(bin(self.p_child[i][0].detach().cpu().numpy()))
        # self.p_child = self.p_child.view(2*self.n_pairs,self.n_params)          # Format the shape

    #########################################################################################################################
    # Integer to binary from :https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits #
    #########################################################################################################################
    def dec2bin(self, x, bits):
        # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
        mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()

    def bin2dec(self, b, bits):
        mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
        return torch.sum(mask * b, -1)

    def float_to_binary(self):
        """
        Convert the floating point number to a binary number
        """
        # First we would multiply the value by INT_NUM then cast to integer
        # print(self.p_child[0])
        self.p_child += 1
        # print(self.p_child.size())
        # print(self.p_child[0])
        self.p_child = torch.max(self.p_child, torch.zeros_like(self.p_child))
        # print(self.p_child.size())
        self.p_child *= INT_NUM
        # print(self.p_child[0])
        self.p_child = self.p_child.type(torch.int32)
        self.p_child = self.dec2bin(self.p_child, NUM_BITS)
        # print(self.p_child.size())
        self.p_child = self.p_child.view(self.n_pairs, 2, self.n_params*NUM_BITS)
        # print(self.p_child[0])
        # print(self.p_child.size())
        # quit()

        
    def binary_to_float(self):
        """
        Convert the binary number into a floating point number
        """
        self.p_child = self.p_child.view(self.n_pairs, 2, self.n_params,NUM_BITS)
        self.p_child = self.bin2dec(self.p_child, NUM_BITS)
        # print(self.p_child[0])
        self.p_child = self.p_child.type(torch.float32)
        self.p_child /= INT_NUM
        # print(self.p_child[0])
        self.p_child -= 1
        # print(self.p_child[0])
        self.p_child = self.p_child.view(self.n_pairs*2, self.n_params)
        # print(self.generation.size())
        # print(self.p_child.size())
        # quit()

    def evolve(self):
        ' Function does the genetic algorithm. It evaluates the next generation given previous one'
        if self.target is None:
            raise(Exception('Set target spectra before running the GA'))

        self.old_gen = self.generation.clone()

        # Evaluate fitness of current population
        logit = self.calculate_spectra(self.generation)
        self.fitness = self.loss_fn(logit,self.target)

        # Select parents for mating and sort individuals in generation
        self.fitness, sorter = torch.sort(self.fitness,descending=False)
        self.generation = self.generation[sorter]

        self.selector()
        # print('before getting mutationa and cross')
        # print(self.p_child)

        self.float_to_binary()
        # Do crossover followed by mutation to create children from parents
        self.X_op()
        # print('after cross')
        # print(self.p_child)
        self.mutate()
        # print('after mutation')
        # print(self.p_child)
        self.binary_to_float()
        
        # print('change back to nonbinary')
        # print(self.p_child)
        # Combine children and elites into new generation
        self.generation[self.n_elite:] = self.p_child[:self.n_kids]

        return logit, self.fitness
