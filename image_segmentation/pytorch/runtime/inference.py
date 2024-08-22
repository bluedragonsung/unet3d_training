import numpy as np
from scipy import signal
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go


import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from runtime.distributed_utils import reduce_tensor, get_world_size, get_rank

# output_feature_map_path = "/DATA/soosung/output_feature_map_numpy_quantized/"
# output_feature_map_path = "/DATA/soosung/output_feature_map_numpy/case0/window"
# input_feature_map_numpy = "/DATA/soosung/input_feature_map_numpy/case41/window"
# output_feature_map_path = "/DATA/grpark/NPU_case0"
ofm_path = "/DATA/soosung/unet3d/ofm_q_npy"

def evaluate(flags, model, loader, loss_fn, score_fn, device, epoch=0, is_distributed=False):
    rank = get_rank()
    world_size = get_world_size()
    model.to(device)
    if flags.load_ckpt_path:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(flags.load_ckpt_path, map_location=map_location)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['best_model_state_dict'], strict=False)
        # print(model.state_dict().values().shape)
        # for tensor in model.state_dict().values():
        #     print(tensor.shape)
        if is_distributed:
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                              device_ids=[flags.local_rank],
                                                              output_device=flags.local_rank)

    model.eval()
    
    eval_loss = []
    scores = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, disable=(rank != 0) or not flags.verbose)):
            image, label = batch
            image, label = image.to(device), label.to(device)
            print("--------------------------")
            print("case", i)
            
            if image.numel() == 0:
                continue
            with autocast(enabled=flags.amp):
                output, label = sliding_window_inference(
                    i,
                    inputs=image,
                    labels=label,
                    roi_shape=flags.val_input_shape,
                    model=model,
                    overlap=flags.overlap,
                    mode="gaussian",
                    padding_val=-2.2
                )
                #########################visualize output########################    
                # print("output shape: ", output.shape)
                # for i in range(128):
                #     frame = output[0, :, i, :, :]   #first frame
                #     gray_frame = frame.mean(dim=0)
                #     gray_frame = (gray_frame - gray_frame.min()) / (gray_frame.max() - gray_frame.min())
                #     gray_frame_np = gray_frame.cpu().numpy()

                #     plt.imshow(gray_frame_np, cmap='gray')
                #     plt.title("3D Image Grayscale")
                #     plt.axis('off')
                #     plt.savefig('output_plot.png')
                ################################################################## 
                
                # Accuracy for each test data #####################################
                scores_ = score_fn(output, label)
                eval_loss_ = loss_fn(output, label)

                scores_, eval_loss_ = scores_.cpu().numpy(), float(eval_loss_.cpu().numpy())
                eval_metrics_ = {"kidney": scores_[-2], "tumor": scores_[-1]}
                
                for key in eval_metrics_.keys():
                    print(key, eval_metrics_[key])
                ###################################################################
                
                # onehot encoding result ##########################################
                # print(output.shape)
                # one_hot_output = torch.argmax(output, dim=1, keepdim=True)
                # one_hot_output_np = one_hot_output.cpu().numpy()
                # print(one_hot_output_np.shape)
                
                # # output size(384,256,256)
                # class_idx=1
                # for depth in range(384):
                #     one_hot_image = (one_hot_output_np[0, 0, depth, :, :] == class_idx)

                #     plt.figure(figsize=(10, 7.5)) 
                #     plt.imshow(one_hot_image, cmap='viridis')  
                #     plt.colorbar()  
                #     plt.title('Visualization of a 3D Data Slice')
                #     print("slice number:", depth)  
                #     plt.savefig('output_plot.png')

                # fig = go.Figure(data=go.Volume(
                # x=np.linspace(0, one_hot_output_np.shape[0]-1, one_hot_output_np.shape[0]),  # x 축
                # y=np.linspace(0, one_hot_output_np.shape[1]-1, one_hot_output_np.shape[1]),  # y 축
                # z=np.linspace(0, one_hot_output_np.shape[2]-1, one_hot_output_np.shape[2]),  # z 축
                # value=one_hot_output_np.flatten(),  # 3D 띰이터 플랫화
                # isomin=0.5,  # 최소값 설정 (0.5는 1엝 해당하는 부분만 표시)
                # isomax=1.5,  # 최대값 설정 (1.5는 1엝 해당하는 부분만 표시)
                # opacity=0.1,  # 불투명띄 설정
                # surface_count=15,  # 표면 갯수
                # colorscale='Gray',  # 색생 스케일
                # caps=dict(x_show=False, y_show=False, z_show=False)  # 객 축의 띝을 표시하지 않음
                # ))

                # fig.update_layout(
                #     title="3D Volume Visualization of Binary Data",
                #     width=700,
                #     height=700,
                #     scene=dict(
                #         xaxis_title='X Axis',
                #         yaxis_title='Y Axis',
                #         zaxis_title='Z Axis'
                #     )
                # )
                # fig.show()
                ###################################################################                
                
                eval_loss_value = loss_fn(output, label)
                scores.append(score_fn(output, label))
                eval_loss.append(eval_loss_value)
            del output
            del label

    scores = reduce_tensor(torch.mean(torch.stack(scores, dim=0), dim=0), world_size)
    eval_loss = reduce_tensor(torch.mean(torch.stack(eval_loss, dim=0), dim=0), world_size)
    
    # scores = torch.mean(torch.stack(scores, dim=0), dim=0)
    # eval_loss = torch.mean(torch.stack(eval_loss, dim=0), dim=0)
        

    scores, eval_loss = scores.cpu().numpy(), float(eval_loss.cpu().numpy())
    eval_metrics = {"epoch": epoch,
                    "L1 dice": scores[-2],
                    "L2 dice": scores[-1],
                    "mean_dice": (scores[-1] + scores[-2]) / 2,
                    "eval_loss": eval_loss}

    return eval_metrics


def pad_input(volume, roi_shape, strides, padding_mode, padding_val, dim=3):
    """
    mode: constant, reflect, replicate, circular
    """
    bounds = [(strides[i] - volume.shape[2:][i] % strides[i]) % strides[i] for i in range(dim)]
    bounds = [bounds[i] if (volume.shape[2:][i] + bounds[i]) >= roi_shape[i] else bounds[i] + strides[i]
              for i in range(dim)]
    paddings = [bounds[2] // 2, bounds[2] - bounds[2] // 2,
                bounds[1] // 2, bounds[1] - bounds[1] // 2,
                bounds[0] // 2, bounds[0] - bounds[0] // 2,
                0, 0,
                0, 0]

    return F.pad(volume, paddings, mode=padding_mode, value=padding_val), paddings


def gaussian_kernel(n, std):
    gaussian1D = signal.gaussian(n, std)
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    gaussian3D = np.outer(gaussian2D, gaussian1D)
    gaussian3D = gaussian3D.reshape(n, n, n)
    gaussian3D = np.cbrt(gaussian3D)
    gaussian3D /= gaussian3D.max()
    return torch.from_numpy(gaussian3D)


def sliding_window_inference(z, inputs, labels, roi_shape, model, overlap=0.5, mode="gaussian",
                             padding_mode="constant", padding_val=0.0, **kwargs):
    image_shape = list(inputs.shape[2:])
    dim = len(image_shape)
    strides = [int(roi_shape[i] * (1 - overlap)) for i in range(dim)]

    bounds = [image_shape[i] % strides[i] for i in range(dim)]
    bounds = [bounds[i] if bounds[i] < strides[i] // 2 else 0 for i in range(dim)]
    inputs = inputs[...,
             bounds[0] // 2: image_shape[0] - (bounds[0] - bounds[0] // 2),
             bounds[1] // 2: image_shape[1] - (bounds[1] - bounds[1] // 2),
             bounds[2] // 2: image_shape[2] - (bounds[2] - bounds[2] // 2)]
    labels = labels[...,
             bounds[0] // 2: image_shape[0] - (bounds[0] - bounds[0] // 2),
             bounds[1] // 2: image_shape[1] - (bounds[1] - bounds[1] // 2),
             bounds[2] // 2: image_shape[2] - (bounds[2] - bounds[2] // 2)]

    inputs, paddings = pad_input(inputs, roi_shape, strides, padding_mode, padding_val)

    padded_shape = inputs.shape[2:]
    size = [(inputs.shape[2:][i] - roi_shape[i]) // strides[i] + 1 for i in range(dim)]
    result = torch.zeros(size=(1, 3, *padded_shape), dtype=inputs.dtype, device=inputs.device)
    norm_map = torch.zeros_like(result)
    if mode == "constant":
        norm_patch = torch.ones(size=roi_shape, dtype=norm_map.dtype, device=norm_map.device)
    elif mode == "gaussian":
        norm_patch = gaussian_kernel(roi_shape[0], 0.125*roi_shape[0]).type(norm_map.dtype).to(norm_map.device)

    else:
        raise ValueError("Unknown mode. Available modes are {constant, gaussian}.")
    ######################################### one image #################################################
    # # print(inputs.dtype)
    # # print(roi_shape[0], roi_shape[1], roi_shape[2])
    # # print(size[0], size[1], size[2])
    # # print(strides[0], strides[1], strides[2])
    # count = 0
    # for i in range(0, strides[0] * size[0], strides[0]):
    #     for j in range(0, strides[1] * size[1], strides[1]):
    #         for k in range(0, strides[2] * size[2], strides[2]):
    #             input_data = inputs[...,
    #                 i:(roi_shape[0]+i),
    #                 j:(roi_shape[1]+j),
    #                 k:(roi_shape[2]+k)
    #                 ] #* norm_patch
    #             #print("norm_patch shape:", norm_patch.shape)
    #             min_value = input_data.min()
    #             input = torch.clamp(input_data + min_value.abs(), min=0.0)
    #             # print("min value of input:", input_data.min())
    #             np.save("/DATA/soosung/feature_map_numpy/input_data" + str(count), input.detach().cpu().numpy())

    #             # print("input data shape:", input_data.shape)
    #             output_data = model(input_data)
    #             np.save("/DATA/soosung/feature_map_numpy/output_data" + str(count), output_data.detach().cpu().numpy())
    #             print("saved window", count)
    #             count = count + 1
    #####################################################################################################    

    ######################################### whole tensor before model #################################
    # print(inputs.dtype)
    # window_num = 0
    # for i in range(0, strides[0] * size[0], strides[0]):
    #     for j in range(0, strides[1] * size[1], strides[1]):
    #         for k in range(0, strides[2] * size[2], strides[2]):
    #             input_data = inputs [...,
    #                 i:(roi_shape[0] + i),
    #                 j:(roi_shape[1] + j),
    #                 k:(roi_shape[2] + k)]
                
    #             np.save(input_feature_map_numpy + str(window_num), input_data.detach().cpu().numpy())
    #             window_num += 1
    #####################################################################################################


    ######################################### whole tensor after pytorch model ###################################
    # # print(size[0], size[1], size[2])
    # for i in range(0, strides[0] * size[0], strides[0]):
    #     for j in range(0, strides[1] * size[1], strides[1]):
    #         for k in range(0, strides[2] * size[2], strides[2]):
    #             result[
    #             ...,
    #             i:(roi_shape[0] + i),
    #             j:(roi_shape[1] + j),
    #             k:(roi_shape[2] + k)] += model(inputs[
    #                                            ...,
    #                                            i:(roi_shape[0] + i),
    #                                            j:(roi_shape[1] + j),
    #                                            k:(roi_shape[2] + k)
    #                                            ]) * norm_patch
    #             # model_output = model(inputs[
    #             #                                ...,
    #             #                                i:(roi_shape[0] + i),
    #             #                                j:(roi_shape[1] + j),
    #             #                                k:(roi_shape[2] + k)
    #             #                                ]) * norm_patch
    #             # #print("Model output at position (i={}, j={}, k={}):".format(i,j,k), model_output)
    #             # print(model_output.shape)
    #             # print(norm_patch)
    #             norm_map[
    #             ...,
    #             i:(roi_shape[0] + i),
    #             j:(roi_shape[1] + j),
    #             k:(roi_shape[2] + k)] += norm_patch
    #####################################################################################################

    ######################################### whole tensor after C model ###################################
    window_num = 0
    print(size[0], size[1], size[2])
    for i in range(0, strides[0] * size[0], strides[0]):
        for j in range(0, strides[1] * size[1], strides[1]):
            for k in range(0, strides[2] * size[2], strides[2]):
                
                file_path = f"{ofm_path}/case{z}/window{window_num}.npy"
                data_npy = np.load(file_path)
                data = torch.from_numpy(data_npy).to(norm_map.device)

                result[
                ...,
                i:(roi_shape[0] + i),
                j:(roi_shape[1] + j),
                k:(roi_shape[2] + k)] += data * norm_patch

                window_num += 1

                norm_map[
                ...,
                i:(roi_shape[0] + i),
                j:(roi_shape[1] + j),
                k:(roi_shape[2] + k)] += norm_patch
    #####################################################################################################

    # account for any overlapping sections
    # norm_map[norm_map == 0] = norm_map[norm_map > 0].min()
    result /= norm_map

    return result[
           ...,
           paddings[4]: image_shape[0] + paddings[4],
           paddings[2]: image_shape[1] + paddings[2],
           paddings[0]: image_shape[2] + paddings[0]
           ], labels
