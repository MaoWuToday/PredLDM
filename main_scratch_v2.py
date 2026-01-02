"""This is the main function for experiments with Benchmark
Precipitation Nowcasting (BPN)."""
import os
import utils
import models
import dataset
import torch as t
from torch import nn
from configs import configs
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# additonally
# from utils.convlstmnet import ConvLSTMNet
from utils.loss_functions import weighted_l2_loss_radar
import sys
from models.constrain_moments import K2M
import numpy as np

import model_dict

from opensora.acceleration.parallel_states import get_sequence_parallel_group
from opensora.schedulers.iddpm import IDDPM
from mmengine.runner import set_random_seed
from einops import rearrange
#
in_len = configs.in_len
out_len = configs.out_len


def ini_model_params(model, ini_mode='xavier'):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.Linear)):
            if ini_mode == 'xavier':
                nn.init.xavier_normal_(m.weight)
            elif ini_mode == 'orthogonal':
                nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)


def test(test_dataloader=None, mode='weather', model_name=None, dBZ_threshold=10):
    set_random_seed(seed=1024)  # for PredLDM
    t.set_grad_enabled(False)
    device = t.device('cuda')

    # build and load vae from pretrained
    """Load model"""
    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' Model structure: \t {}'.format(configs.model))
    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " -- Prepare vae with AutoencoderKL")
    vae = models.VideoAutoencoderKL(from_pretrained='stabilityai/sd-vae-ft-ema',
                                    micro_batch_size=4, cache_dir='cached_models').to(device).eval()
    # vae = models.VideoAutoencoderKL(from_pretrained=None,
                                    # micro_batch_size=4, cache_dir='cached_models').to(device).eval()
    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " -- Vae is built and parameters are loaded")
    
    vae_name = configs.model_save_dir + '/' + configs.pretrained_model + 'vae.pth'
    vae.load_state_dict(t.load(vae_name))
    
    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "  Start Test")

    # get the size of tensors for diffusions
    # Model setting
    """Load in_len, out_len, shape"""
    in_len, out_len = dataset.get_len(configs)
    img_width, img_height, channel_num = dataset.get_shape(configs)
    log_dir = os.path.join(configs.test_imgs_save_dir, configs.dataset_type)
    if configs.dataset_type == 'pam':
        log_dir = os.path.join(log_dir, configs.domain)
    else:
        print('error in save imgs path')
    log_dir = os.path.join(log_dir, configs.pretrained_model)

    channel_num = 3

    num_frames = in_len
    image_size = (img_width, img_height)
    input_size = (num_frames, *image_size)  # make sure is (10, 64, 64)
    latent_size = vae.get_latent_size(input_size)
    vae_out_channels = vae.out_channels
    print("Time: " + datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S') + " -- The latent size is: {} and the vae out channels is: {}.".format(latent_size,
                                                                                                    vae_out_channels))

    # build latnet diffusions (currently we use the samll scale of parameters), directly load from pretrained (optional) or load from self-trained.
    model = models.STDiT_XL_2(input_size=latent_size,
                              in_channels=vae_out_channels,
                              space_scale=0.5,
                              time_scale=1.0,
                              # from_pretrained="PixArt-XL-2-512x512.pth",   # no efffect
                              # from_pretrained="PredLDM_epoch60.pth",    #  wrong usage of opensora load_checkpoint
                              enable_flash_attn=False,    # for temp
                              enable_layernorm_kernel=False,
                              ).to(device).eval()
    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " -- Diffusions are built and loaded")
    model_name = configs.model_save_dir + '/' + configs.pretrained_model + '.pth'
    model.load_state_dict(t.load(model_name))
    
    
    scheduler = IDDPM(num_sampling_steps=1000, cfg_scale=7.0, )  # may be should be 1000
    # scheduler = IDDPM(num_sampling_steps=1000, cfg_scale=7.0, )

    # basic settings
    fps = 24 // 3
    save_fps = 1
    multi_resolution = None
    batch_size = 1
    num_sample = 1
    loop = 1
    verbose = 1
    masks = None

    """Load dataloader"""
    if test_dataloader == None:
        _, _, test_dataloader = dataset.load_data(configs)
    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
          'Load dataset successfully...')

    if model_name == None:
        model_name = configs.model_save_dir + '/' + configs.pretrained_model + '.pth'
    else:
        print('Testing model: {}'.format(model_name))

    # if configs.use_gpu:
    #     device = t.device('cuda:0')
    #     if len(configs.device_ids_eval) > 1:
    #         model = nn.DataParallel(model, device_ids=configs.device_ids_eval, dim=0)
    #         model.to(device)
    #     else:
    #         model.to(device)
    # print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "  Load successfully {}".format(model_name))

    # indexes
    model.eval()
    if mode == 'weather':
        pod, far, csi, bias, hss, index, ssim, score = [], [], [], [], [], [], [], []
    elif mode == 'simple':
        mse, psnr, ssim, lpips = [], [], [], []
    
    index = []
    
    with t.no_grad():
        l_t = len(test_dataloader.dataset)
        fq = round(l_t / 100)
        i = 0
        for iter, data in enumerate(test_dataloader):
            # if iter >= 10:
                # break
            # if iter > 120:
                # break
            # if iter != 120:
                # continue
            try:
                if iter % fq == 0:
                    i += 1
            except:
                print('111')
            # print(iter%fq, l_t)
            print("\r", end="")
            print("valid progress: {}%: ".format(i), "▋" * (i // 2), end="")

            # Address data from dataloader
            input = data[:, 0:in_len].to(device)
            ground_truth = data[:, in_len:(in_len + out_len)].to(device)
            
            if configs.model == 'PredLDM' and input.size(2) == 1:
                input = input.repeat(1, 1, 3, 1, 1)
                ground_truth = ground_truth.repeat(1, 1, 3, 1, 1)
            
             # for rearrange the channels for link vae
            input = rearrange(input, "B T C W H -> B C T W H")
            ground_truth = rearrange(ground_truth, "B T C W H -> B C T W H")
            
            # check if the input len and out len are not ==, if not continue, for caltefch datset, it happens sometimes
            if input.size()[2] != ground_truth.size()[2]:
                continue
            print('first')
            print(input.shape, ground_truth.shape)   # here is okay, [B C T W H], match vae, and consistent

            # Prepare output (generating output on input) and ground-truth
            # ground_truth = input
            batch_prompts_loop = input
            if configs.model == 'PredLDM':
                try:
                    # == Iter over loop generation ==
                    print('iter over loop')
                    video_clips = []
                    for loop_i in range(loop):
                        # == sampling ==
                        t.manual_seed(1024)
                        # z = torch.randn(len(batch_prompts), vae_out_channels, *latent_size, device=device, dtype=dtype)   # dtype = "bf16"
                        z = t.randn(1, vae_out_channels, *latent_size, device=device)
                        encoder = vae
                        model_args = None
                        samples = scheduler.sample(
                            model,
                            encoder,
                            z=z,
                            prompts=batch_prompts_loop,
                            device=device,
                            additional_args=model_args,
                            progress=verbose >= 2,
                            mask=masks,
                        )
                        # samples = vae.decode(samples.to(dtype), num_frames=num_frames)
                        samples = vae.decode(samples, num_frames=num_frames)
                        video_clips.append(samples)
                    # print(video_clips)
                except RuntimeError as e:
                    print('Runtime Error ' + str(e))
            
            
            
            print(len(video_clips))
            idx = 0
            batch_prompt = None
            video = [video_clips[i][idx] for i in range(loop)]
            print(len(video))
            # batch_prompts_loop
            video = t.cat(video, dim=1)
            print(video.shape)
            output = video
            # here output the shape is [c t w h], but the next dims need [b t c w h], as training is no need, so no problem for training.
            # for rearrange the channels for link next
            output = t.unsqueeze(output, dim=0)    # change to -> [1, c, t, w, h], cnm, code online trick me
            print(output.shape)
            output = rearrange(output, "B C T W H -> B T C W H")
            ground_truth = rearrange(ground_truth, "B T C W H -> B C T W H")
            input = rearrange(input, "B C T W H -> B T C W H")
            print('second')
            print(input.shape, output.shape, ground_truth.shape)
            
            # if configs.model == 'MS2Pv3':
            #     output = output[0]
            
            # TEMP ADDED
            print(configs.save_open, input.shape, output.shape, ground_truth.shape)
            if configs.save_open:
                print('saving.... iter {} output shape {}'.format(iter, output.shape))
                utils.save_test_imgs(log_dir, iter, input, output, ground_truth,
                                     configs.dataset_type, save_mode=configs.save_mode)
            
#             if mode == 'weather':
#                 pod_, far_, csi_, bias_, hss_, index_ = utils.crosstab_evaluate(output, ground_truth, dBZ_threshold, 70,
#                                                                                 dataset)
#             elif mode == 'simple':
#                 # mse_, psnr_, lpips_ = utils.crosstab_evaluate_simple(output, ground_truth)
#                 try:
#                     mse_, psnr_, lpips_ = utils.crosstab_evaluate_simple(output, ground_truth)
#                 except:
#                     print('something wrong, if for kitticaltech, the len of in and out is not ok in 1st sample test.')
#                     continue

#             if mode == 'weather':
#                 pod.append(pod_.data)
#                 far.append(far_.data)
#                 csi.append(csi_.data)
#                 bias.append(bias_.data)
#                 hss.append(hss_.data)
#                 index.append(index_)
#                 ssim_ = utils.compute_ssim(output.cpu().numpy(), ground_truth.cpu().numpy())
#                 ssim.append(ssim_)
#             elif mode == 'simple':
#                 mse.append(mse_)
#                 psnr.append(psnr_)
#                 lpips.append(lpips_)
#                 ssim_ = utils.compute_ssim(output.cpu().numpy(), ground_truth.cpu().numpy())
#                 ssim_ = ssim_.cpu().numpy()
#                 ssim.append(ssim_)
#             # print(index_.size())
            
#             print(mse_)
#             print(psnr_)
#             print(ssim_)
            
#             # save seqs ?
#             print(configs.save_open)
#             if configs.save_open:
#                 print('saving.... iter {} output shape {}'.format(iter, output.shape))
#                 utils.save_test_imgs(log_dir, iter, input, output, ground_truth,
#                                      configs.dataset_type, save_mode=configs.save_mode)
                
#             index.append(iter)
#             # if iter >= 50:
#             #     break
#         # index = t.cat(index, dim=1)  # not appropriate for ours
#         print(ssim)
#         print('the averaged ....')
#         ssim_t = [np.mean(arr[0]) for arr in ssim]
#         print(ssim_t)
#         print(index)
        
#         sorted_indices = np.argsort(ssim_t)[::-1]
        
#         ssim_t = np.array(ssim_t)
        
#         # 获取最大的100个元素的索引
#         top_100_indices = sorted_indices[:1000]
#         print("B中前100个最大元素的索引:", top_100_indices)

#         # 获取这些索引在A中的对应值
#         top_100_A_indices = [index[i] for i in top_100_indices]
#         print("这些索引在A中的对应值:", top_100_A_indices)

#         # 获取B中前100个最大元素的数值
#         top_100_values = ssim_t[top_100_indices]
#         print("B中前100个最大元素的数值:", top_100_values)
        
#         meanv = np.mean(top_100_values)
#         print('mean value ssim', meanv)
        
#         # 打印结果
        
        
        
        
        
        
#         if mode == 'weather':
#             index = t.cat(index, dim=0)
#             data_num = index.numel()
#             # the ground-truth sample which has no rainfall preddiction hits will not be included in calculation
#             # cal_num = index.size()[1] - t.sum(index, dim=1) if eval_by_seq is True else data_num - t.sum(index)  # not apppri...
#             # print(cal_num)
#             cal_num = data_num - t.sum(index)
#             # print(t.sum(index))

#             pod = out_len * t.sum(t.cat(pod, dim=0), 0) / cal_num
#             far = out_len * t.sum(t.cat(far, dim=0), 0) / cal_num
#             csi = out_len * t.sum(t.cat(csi, dim=0), 0) / cal_num
#             bias = out_len * t.sum(t.cat(bias, dim=0), 0) / cal_num
#             hss = out_len * t.sum(t.cat(hss, dim=0), 0) / cal_num
#             ssim = t.mean(t.cat(ssim, dim=0), 0)

#             # pod_sum = t.sum(t.cat(pod, dim=0)) / cal_num
#             # far_sum = t.sum(t.cat(far, dim=0)) / cal_num
#             # csi_sum = t.sum(t.cat(csi, dim=0)) / cal_num
#             # bias_sum = t.sum(t.cat(bias, dim=0)) / cal_num
#             # hss_sum = t.sum(t.cat(hss, dim=0)) / cal_num
#             # ssim_sum = t.mean(t.cat(ssim, dim=0))
#         elif mode == 'simple':
#             mse = np.average(mse, axis=0)
#             psnr = np.average(psnr, axis=0)
#             ssim = np.average(ssim, axis=0)
#             lpips = np.average(lpips, axis=0)
#             # pass
        
#     # model.train()
#     if mode == 'weather':
#         print('\n pod: {}\n far: {}\n csi: {}\n bias: {}\n hss: {}\n ssim: {}\n'.format(pod, far, csi, bias, hss, ssim))
#         print('Time: ' + datetime.now().strftime(
#             '%Y-%m-%d %H:%M:%S') + '  Test:\tPOD: {:.4f}, FAR: {:.4f}, CSI: {:.4f}, BIAS: {:.4f}, HSS: {:.4f}, SSIM: {:.4f}'
#               .format(t.mean(pod), t.mean(far), t.mean(csi), t.mean(bias), t.mean(hss), t.mean(ssim)))
#     elif mode == 'simple':
#         print('mse: {}\n psnr: {}\n ssim: {}\n lpips: {}\n'.format(mse, psnr, ssim, lpips))
#         print('Time: ' + datetime.now().strftime(
#             '%Y-%m-%d %H:%M:%S') + '  Test:\tMSE: {:.4f}, PSNR: {:.4f}, SSIM: {:.4f}, LPIPS: {:.4f}'
#               .format(np.average(mse), np.average(psnr), np.average(ssim), np.average(lpips)))

#     # if not os.path.exists(configs.test_imgs_save_dir):
#     #     os.makedirs(configs.test_imgs_save_dir)
#     # if mode == 'weather':
#     #     utils.save_test_results(configs.test_imgs_save_dir, pod, far, csi, bias, hss, ssim)
#     # elif mode == 'simple':
#     #     utils.save_test_results(configs.test_imgs_save_dir, mse, psnr, ssim, lpips)

#     if mode == 'weather':
#         return pod, far, csi, bias, hss, ssim
#     elif mode == 'simple':
#         return mse, psnr, ssim, lpips
    
    retrun


def train():
    device = t.device('cuda')

    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Train mode is a go. ")
    """Load in_len, out_len, shape"""
    in_len, out_len = dataset.get_len(configs)
    img_width, img_height, channel_num = dataset.get_shape(configs)
    
    # in_len, out_len = 10, 10
    
    """Load model"""
    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' Model structure: \t {}'.format(configs.model))
    # build and load vae from pretrained
    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " -- Prepare vae with AutoencoderKL")
    vae = models.VideoAutoencoderKL(from_pretrained='stabilityai/sd-vae-ft-ema',
                                    micro_batch_size=None,
                                    cache_dir='cached_models',
                                    ).to(device)  # the micro_batch_size can be no effect on time comsuming
    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " -- Vae is built and parameters are loaded")

    # get the size of tensors for diffusions
    # input_size = (dataset.num_frames, *dataset.image_size)
    latent_size = vae.get_latent_size(
        (in_len, img_width,
         img_height))  # (10 8 8) acutally the input size is (10, 64, 64), after load params, it becomes (10, 8, 8)
    vae_out_channels = vae.out_channels  # ok for 4.
    print("Time: " + datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S') + " -- The latent size is: {} and the vae out channels is: {}.".format(latent_size,
                                                                                                    vae_out_channels))

    # build latnet diffusions (currently we use the samll scale of parameters), load from pretrained (optional)
    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " -- Prepare diffusions")
    model = models.STDiT_XL_2(input_size=latent_size,
                              in_channels=vae_out_channels,
                              space_scale=0.5,
                              time_scale=1.0,
                              # from_pretrained="PixArt-XL-2-512x512.pth",
                              enable_flash_attn=False,    # as installation problem, temporally disabled
                              enable_layernorm_kernel=False,
                              ).to(device)
    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " -- Diffusions are built")

    # freeze the vae and let diffusions be trainable
    # optimizer = t.optim.Adam(model.parameters(), lr=configs.learning_rate, betas=configs.optim_betas)
    # scheduler = t.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    optimizer = t.optim.Adam(
        [{'params': param} for param in model.parameters() if param.requires_grad],
        lr=1e-4,
        betas=(0.9, 0.999),
    )
    scheduler = t.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1.0)

    # Pre-training or tuning setting
    if configs.fine_tune:
        print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
              'Fine tuning based on pre-trained model: {}.'.format(configs.pretrained_model))
        # contune training
        model.load_state_dict(t.load(configs.model_save_dir + '/' + configs.pretrained_model + '.pth'))
    else:
        # print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
        #       'Learning from scratch. Initializing the model params...')
        # ini_model_params(vae, configs.ini_mode)
        ini_model_params(model, configs.ini_mode)
        print("Time: " + datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S') + ' -- Initialize with mode. if load pretrained parameters, need to change.')
    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "  Load Model Successfully")

    # # GPU setting
    # if configs.use_gpu:
    #     print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' Using GPU... ids: \t{}'.format(
    #         str(configs.device_ids)))
    #     device = t.device('cuda:0')
    #     if len(configs.device_ids) > 1:
    #         model = nn.DataParallel(model, device_ids=configs.device_ids, dim=0)
    #         model.to(device)
    #     else:
    #         # model.to(device).double()
    #         model.to(device)
    # else:
    #     print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + 'Using CPU...')

    """Load dataloader"""
    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " -- Load dataset")
    train_dataloader, valid_dataloader, test_dataloader = dataset.load_data(configs)
    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
          'Load dataset successfully...')

    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "  Start Train")
    # train_global_step = 0

    """Training in epochs"""
    for epoch in range(configs.train_max_epoch):
        # epoch += 32
        """Training"""
        for iter, data in enumerate(train_dataloader):
            # randomly sampling
            if configs.random_sampling:
                if iter * configs.batch_size >= configs.random_iters:
                    break

            # Address data from dataloader
            if configs.dataset_type == 'pam':
                pre_seq = data[:, 0:in_len].to(device)
                fut_seq = data[:, in_len:(in_len + out_len)].to(device)
            else:
                print('The dataset type is error.')
            optimizer.zero_grad()
            
            # for rearrange the channels for link vae
            pre_seq = rearrange(pre_seq, "B T C W H -> B C T W H")
            fut_seq = rearrange(fut_seq, "B T C W H -> B C T W H")
            
            # vae encoding the future sequences and encoding conditions (previous sequence)
            x = vae.encode(fut_seq)  # [B, C, T, H/P, W/P]
            y = vae.encode(pre_seq)
            model_args = dict(y=y)  # model_args = dict(y=torch.randn(2, 3, 10, 64, 64))

            lossitem = IDDPM()
            loss_dict = lossitem.training_losses(model=model, x_start=x, model_kwargs=model_args)
            loss = loss_dict["loss"].mean()

            loss.backward()
            optimizer.step()
            scheduler.step()
            # train_global_step += 1

            # Print loss & log loss
            if (iter + 1) % configs.train_print_fre == 0:
                print('Time: ' + datetime.now().strftime(
                    '%Y-%m-%d %H:%M:%S') + '  Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format((epoch + 1), (
                        iter + 1) * configs.batch_size, len(train_dataloader) * configs.batch_size, 100. * (
                                                                                                              iter + 1) / len(
                    train_dataloader), loss.item()))
            # break

        # Save by epochs
        if not os.path.exists(configs.model_save_dir):
            os.makedirs(configs.model_save_dir)

        if (epoch + 1) % configs.model_save_fre == 0:
            model_name = configs.model_save_dir + '/' + configs.model + '_epoch' + str(epoch + 1) + '.pth'
            vae_name = configs.model_save_dir + '/' + configs.model + '_epoch' + str(epoch + 1) + 'vae.pth'
            if len(configs.device_ids) > 1:
                t.save(model.module.state_dict(), model_name)
                t.save(vae.module.state_dict(), vae_name)
            else:
                t.save(model.state_dict(), model_name)
                t.save(vae.state_dict(), vae_name)


def main():
    if configs.mode == 'train':
        train()
    elif configs.mode == 'test':
        test(mode=configs.eval_mode)


if __name__ == '__main__':
    main()

    