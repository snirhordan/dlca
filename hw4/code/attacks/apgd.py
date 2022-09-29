import numpy as np
import torch
from attacks.attack import Attack
import time
from tqdm import tqdm
import cv2
from .noiser import Noise

class APGD(Attack):	
    def __init__(	
            self,	
            model,	
            criterion,	
            test_criterion,	
            data_shape,	
            norm='Linf',	
            n_iter=20,	
            n_restarts=1,
            alpha=None,	
            rand_init=False,	
            stochastic=False,	
            sample_window_size=None,	
            sample_window_stride=None,	
            frames_exp_factor=0,	
            pert_padding=(0, 0),	
            init_pert_path=None,	
            init_pert_transform=None):	
        super(APGD, self).__init__(model, criterion, test_criterion, norm, data_shape, stochastic,	
                                  sample_window_size, sample_window_stride, frames_exp_factor,	
                                  pert_padding)	

        self.alpha = alpha

        self.n_restarts = n_restarts
        self.n_iter = n_iter

        self.rand_init = rand_init

        self.init_pert = None
        if init_pert_path is not None:
            self.init_pert = cv2.cvtColor(cv2.imread(init_pert_path), cv2.COLOR_BGR2RGB)
            if init_pert_transform is None:
                self.init_pert = torch.tensor(self.init_pert).unsqueeze(0)
            else:
                self.init_pert = init_pert_transform({'img': self.init_pert})['img'].unsqueeze(0)
    def sampler(self, pert, height, width, eps, device=None) -> torch.tensor:
        height = int(height)
        print("width is:")
        print("{}".format(str(width)))
        patch = torch.zeros_like(pert.clone().detach()).to(device)
        rand_i = torch.randint(0, int(width) - int(height), (1,)).item()
        rand_i = int(rand_i)
        rand_j = torch.randint(0, int(width) - int(height), (1,)).item()
        rand_j = int(rand_j)
        for color in range(3):
            color=int(color)
            update = 2*eps*torch.bernoulli(torch.tensor([0.5]))[0].to(device)
            patch[0,color,rand_i:rand_i + height,rand_j:rand_j + height] += update * torch.ones((height,height)).to(device)
        return patch
    def eps_project(self, eps, x, x_hat):
        return torch.clamp(x_hat, x-eps, x+eps)
    def project(self, pert, eps):
        return torch.clamp(pert, 1-eps, eps)
    def calc_sample_grad_single(self, pert, img1_I0, img2_I0, intrinsic_I0, img1_delta, img2_delta,
                         scale, y, clean_flow, target_pose, perspective1, perspective2, mask1, mask2, device=None):
        pert = pert.detach()
        pert.requires_grad_()
        img1_adv, img2_adv, output_adv = self.perturb_model_single(pert, img1_I0, img2_I0,
                                                            intrinsic_I0,
                                                            img1_delta, img2_delta,
                                                            scale,
                                                            mask1, mask2,
                                                            perspective1,
                                                            perspective2,
                                                            device)
        loss = self.criterion(output_adv, scale.to(device), y.to(device), target_pose.to(device), clean_flow.to(device))
        #sm = torch.nn.Softmax()#change to weighted sum, more weight onto the latest layer, maybe softmax
        #loss = sm(loss) 
        loss_sum = loss.sum(dim=0)
        grad = torch.autograd.grad(loss_sum, [pert])[0].detach()

        del img1_adv
        del img2_adv
        del output_adv
        del loss
        del loss_sum
        torch.cuda.empty_cache()

        return grad

    def calc_sample_grad_split(self, pert, img1_I0, img2_I0, intrinsic_I0, img1_delta, img2_delta,
                         scale, y, clean_flow, target_pose, perspective1, perspective2, mask1, mask2, device=None):
        sample_data_ind = list(range(img1_I0.shape[0] + 1))
        window_start_list = sample_data_ind[0::self.sample_window_stride]
        window_end_list = sample_data_ind[self.sample_window_size::self.sample_window_stride]

        if window_end_list[-1] != sample_data_ind[-1]:
            window_end_list.append(sample_data_ind[-1])
        grad = torch.zeros_like(pert, requires_grad=False)
        grad_multiplicity = torch.zeros(grad.shape[0], device=grad.device, dtype=grad.dtype)

        for window_idx, window_end in enumerate(window_end_list):
            window_start = window_start_list[window_idx]
            grad_multiplicity[window_start:window_end] += 1

            pert_window = pert[window_start:window_end].clone().detach()
            img1_I0_window = img1_I0[window_start:window_end].clone().detach()
            img2_I0_window = img2_I0[window_start:window_end].clone().detach()
            intrinsic_I0_window = intrinsic_I0[window_start:window_end].clone().detach()
            img1_delta_window = img1_delta[window_start:window_end].clone().detach()
            img2_delta_window = img2_delta[window_start:window_end].clone().detach()
            scale_window = scale[window_start:window_end].clone().detach()
            y_window = y[window_start:window_end].clone().detach()
            clean_flow_window = clean_flow[window_start:window_end].clone().detach()
            target_pose_window = target_pose.clone().detach()
            perspective1_window = perspective1[window_start:window_end].clone().detach()
            perspective2_window = perspective2[window_start:window_end].clone().detach()
            mask1_window = mask1[window_start:window_end].clone().detach()
            mask2_window = mask2[window_start:window_end].clone().detach()

            grad_window = self.calc_sample_grad_single(pert_window,
                                                     img1_I0_window,
                                                     img2_I0_window,
                                                     intrinsic_I0_window,
                                                     img1_delta_window,
                                                     img2_delta_window,
                                                     scale_window,
                                                     y_window,
                                                     clean_flow_window,
                                                     target_pose_window,
                                                     perspective1_window,
                                                     perspective2_window,
                                                     mask1_window,
                                                     mask2_window,
                                                     device=device)
            with torch.no_grad():
                grad[window_start:window_end] += grad_window

            del grad_window
            del pert_window
            del img1_I0_window
            del img2_I0_window
            del intrinsic_I0_window
            del scale_window
            del y_window
            del clean_flow_window
            del target_pose_window
            del perspective1_window
            del perspective2_window
            del mask1_window
            del mask2_window
            torch.cuda.empty_cache()
        grad_multiplicity_expand = grad_multiplicity.view(-1, 1, 1, 1).expand(grad.shape)
        grad = grad / grad_multiplicity_expand
        del grad_multiplicity
        del grad_multiplicity_expand
        torch.cuda.empty_cache()
        return grad.to(device)

    def perturb(self, data_loader, y_list, eps,
                                   targeted=False, device=None, eval_data_loader=None, eval_y_list=None):

        #a_abs = np.abs(eps / self.n_iter) if self.alpha is None else np.abs(self.alpha)
        
        multiplier = -1 if targeted else 1
        print("computing PGD attack with parameters:")
        print("attack random restarts: " + str(self.n_restarts))
        print("attack epochs: " + str(self.n_iter))
        print("attack norm: " + str(self.norm))
        print("attack epsilon norm limitation: " + str(eps))
        #print("attack step size: " + str(a_abs))
        noiser = Noise()
        data_shape, dtype, eval_data_loader, eval_y_list, clean_flow_list, \
        eval_clean_loss_list, traj_clean_loss_mean_list, clean_loss_sum, \
        best_pert, best_loss_list, best_loss_sum, all_loss, all_best_loss = \
            self.compute_clean_baseline(data_loader, y_list, eval_data_loader, eval_y_list, device=device)
        ##############################################################################
        from torch import divide
        #######################eta arguments #########################################

        w = [] #keep as list
        p_minus_two = 0
        w.append( 0 ) 
        p_minus_one = 0.22
        w_1 = torch.ceil( torch.tensor([p_minus_one * self.n_iter], dtype=torch.float))[0]
        w.append( w_1 )
        while divide(torch.ceil( torch.tensor([p_minus_one *self.n_iter], dtype=torch.float)[0] ),self.n_iter) <= 1:
            values = torch.tensor([p_minus_one - p_minus_two - 0.08, 0.03])
            change = torch.max(values)
            p = p_minus_one + change
            w_1 = torch.ceil(torch.tensor([p * self.n_iter], dtype=torch.float)[0]  )
            w.append( w_1 )
            p_minus_two = p_minus_one
            p_minus_one =p
        rho = 0.75 # as mentioned in paper
        a_abs = 0.02
        counter_increase_loss = 0 #condition 1, halving
        loss_list = [(a_abs, 0)]    #condition 2, prevents getting stuck, save (eta, f)
        print("learning rate update vector is")
        print(" " + str(w))
        ##############################################################################
        
        for rest in tqdm(range(self.n_restarts)):
            print("restarting attack optimization, restart number: " + str(rest))
            opt_start_time = time.time()

            pert = torch.zeros_like(best_pert)

            if self.init_pert is not None:
                print(" perturbation initialized from provided image")
                pert = self.init_pert.to(best_pert)
            elif self.rand_init:
                print(" perturbation initialized randomly")
                pert = self.random_initialization(pert, eps)
            else:
                print(" perturbation initialized to zero")

            pert = self.project(pert, eps)
            x_t_minus_two = torch.zeros_like(pert).detach()
            x_t_minus_one = torch.zeros_like(pert).detach()
            x_t = torch.clone(pert).detach()
            #small = 0.01
            for k in tqdm(range(self.n_iter)):
                print(" attack optimization epoch: " + str(k))
                iter_start_time = time.time()
                
                z = self.a_gradient_ascent_step(x_t, data_shape, data_loader, y_list, clean_flow_list,
                                        multiplier, a_abs, eps, device=device)
                
                step_runtime = time.time() - iter_start_time
                print(" optimization epoch finished, epoch runtime: " + str(step_runtime))

                print(" evaluating perturbation")
                eval_start_time = time.time()

                with torch.no_grad():
                    if k % 3 == 0:
                         z = noiser(z, device)#add Gaussian noise
                    if self.stochastic:
                        x_t  = z.clone().detach()
                        #torch.save(x_t, 'tensor_{}.pt'.format(str(k)))
                        print("shape of x_t is :")
                        print("{}".format(str(list(x_t.size()))))
                        width = x_t.size(dim=2)
                        eps = 0.07
                        p = 0.09
                        if k % 100 == 0:
                            p = p/2
                        vari = torch.tensor([p*(width**2)])
                        height = torch.ceil(torch.sqrt( vari )).item()
                        print("height is :")
                        print("{}".format(str(height)))
                        x_init = x_t.clone().detach()
                        eval_loss_tot, eval_loss_list_init = self.attack_eval(x_t, data_shape, eval_data_loader, eval_y_list,
                                             device)
                        loss_init = eval_loss_tot
                        counter = 0
                        while eval_loss_tot <= loss_init and counter <= 15 and k % 100 ==0:
                            added = self.sampler(x_t, height, width, eps, device)
                            x_t = self.eps_project( eps, x_init, added )
                            x_t = self.project(x_t, eps=0)
                            eval_loss_tot, eval_loss_list = self.attack_eval(x_t, data_shape, eval_data_loader, eval_y_list,
                                             device)
                            counter+=1
                        if eval_loss_tot <= loss_init:
                            eval_loss_tot = loss_init
                            eval_loss_list = eval_loss_list_init
                            x_t = torch.clone(x_init).detach()

                    if not self.stochastic:
                        # update momentum inspired update
                        alpha = 0.75 # as in paper
                        momentum = torch.mul(( z - x_t_minus_one ),alpha) + torch.mul((x_t_minus_one - x_t_minus_two),(1-alpha))
                        #momentum = self.normalize_grad(momentum) # not sure necessary
                        x_t = self.project(x_t_minus_one + momentum, eps)
                        x_t_minus_two = x_t_minus_one
                        x_t_minus_one = x_t
                        eval_loss_tot, eval_loss_list = self.attack_eval(x_t, data_shape, eval_data_loader, eval_y_list,
                                                                     device)

                    if eval_loss_tot > best_loss_sum:
                        best_pert = x_t.clone().detach()
                        best_loss_list = eval_loss_list
                        best_loss_sum = eval_loss_tot
                        counter_increase_loss += 1

                    if (k in w) and k > 0:
                        index = w.index( k ) #works only for list not torch tensor
                        margin = w[index] - w[index-1]
                        probability = torch.divide(torch.tensor(counter_increase_loss), torch.tensor(margin))
                        if probability < rho: #mitigates not getting out of local minima
                            a_abs = torch.divide(a_abs, 2)
                            x_t = torch.clone(best_pert).detach()
                            print("PICKED BEST!")
                        if loss_list[-1][1] == best_loss_sum:
                            if loss_list[-1][0] == a_abs:
                                a_abs = torch.divide(a_abs, 2)
                                x_t = torch.clone(best_pert).detach()
                        loss_list.append((a_abs, best_loss_sum))
                        counter_increase_loss = 0

                    all_loss.append(eval_loss_list)
                    all_best_loss.append(best_loss_list)
                    traj_loss_mean_list = np.mean(eval_loss_list, axis=0)
                    traj_best_loss_mean_list = np.mean(best_loss_list, axis=0)

                    eval_runtime = time.time() - eval_start_time
                    print(" evaluation finished, evaluation runtime: " + str(eval_runtime))
                    print(" current trajectories loss mean list:")
                    print(" " + str(traj_loss_mean_list))
                    print(" current trajectories best loss mean list:")
                    print(" " + str(traj_best_loss_mean_list))
                    print(" trajectories clean loss mean list:")
                    print(" " + str(traj_clean_loss_mean_list))
                    print(" current trajectories loss sum:")
                    print(" " + str(eval_loss_tot))
                    print(" current trajectories best loss sum:")
                    print(" " + str(best_loss_sum))
                    print(" trajectories clean loss sum:")
                    print(" " + str(clean_loss_sum))
                    print("Learning rate for iteration :")
                    print(" "+ str(a_abs))
                    del eval_loss_tot
                    del eval_loss_list
                    torch.cuda.empty_cache()

            opt_runtime = time.time() - opt_start_time
            print("optimization restart finished, optimization runtime: " + str(opt_runtime))
        return best_pert.detach(), eval_clean_loss_list, all_loss, all_best_loss
