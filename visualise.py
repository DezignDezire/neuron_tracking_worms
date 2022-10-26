import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import colors
from pytorch_lightning.utilities.distributed import rank_zero_only

@rank_zero_only
def visualise_neurons(inputs, targets, predictions, batch, mode, run_name, unusable_points=None, assoc=None):
    gs_kw = dict(height_ratios=[10.3, 1])
    fig, ax = plt.subplots(2, 2, figsize=(20, 10), dpi=300, gridspec_kw=gs_kw)
    fig.suptitle('batch' + str(batch), fontsize=20)

    for i in range(2):
        target_points = targets[i].reshape(109, 3).cpu()
        predictions_points = predictions[i].reshape(109, 3).cpu()
        unusable_i = unusable_points[i] if unusable_points != None else None
        assoc_i = assoc[i] if assoc != None else None
        plot_scatters(ax, i, inputs.cpu(), target_points, predictions_points, unusable_i, assoc_i)
        ##########################
        # plot_usable_points(inputs.cpu(), target_points, run_name)
        ##########################
    plt.savefig('runs/{}/progress_images/{}_{}'.format(run_name, mode, str(batch)))
    plt.close('all')
    # plt.show()


def plot_scatters(ax, i, inputs, target_points, predictions_points, unusable_points, assoc):
    channeled_image_xy = torch.squeeze(torch.mean(inputs[i], dim = 0), dim = 0)
    ax[0,i].imshow(channeled_image_xy, cmap='inferno')

    point_inds = list(range(target_points.shape[0]))
    mask = list(set(torch.where(torch.tensor(target_points) == torch.tensor([0, 0, 0]))[0].numpy()))
    # print("invisible points:", len(mask))
    point_inds = list(filter(lambda x:x not in mask, point_inds))
    target_points = retransform_points(target_points)
    predictions_points = retransform_points(predictions_points)
    
    # plot unusable points
    if unusable_points != None:
        unusable_points = unusable_points[0].cpu().tolist()
        X_cords = [target_points[unusable_points,2].tolist(), predictions_points[unusable_points,2].tolist()]
        Y_cords = [target_points[unusable_points,1].tolist(), predictions_points[unusable_points,1].tolist()]
        ax[0,i].plot(X_cords, Y_cords, color="red", linewidth=0.75)

    ax[0,i].scatter(target_points[point_inds,2], target_points[point_inds,1], c="white", s=1.1)
    ax[0,i].scatter(predictions_points[point_inds,2], predictions_points[point_inds,1], c="black", s=1.5)

    if unusable_points != None:
        ax[0,i].scatter(predictions_points[unusable_points,2], predictions_points[unusable_points,1], c="red", s=1.5)
    
    if assoc != None:
        inds = torch.range(0, len(assoc)-1)
        correct_assoc_inds = torch.where(inds == assoc.cpu())[0].numpy()
        combined_inds = list(set(point_inds).intersection(set(correct_assoc_inds)))
        ax[0,i].scatter(predictions_points[combined_inds,2], predictions_points[combined_inds,1], c="green", s=1.5)
    
    channeled_image_xz = torch.squeeze(torch.mean(inputs[i], dim = 1), dim = 0)
    channeled_image_xz = torch.repeat_interleave(channeled_image_xz, 3, dim=0)
    ax[1,i].imshow(channeled_image_xz, cmap='inferno')
    ax[1,i].scatter(target_points[point_inds,2], target_points[point_inds,0], c="white", s=0.5)
    ax[1,i].scatter(predictions_points[point_inds,2], predictions_points[point_inds,0], c="black", s=0.5)

    x_scale = torch.max(target_points[:,2]) - torch.min(target_points[:,2])
    y_scale = torch.max(target_points[:,1]) - torch.min(target_points[:,1])
    x_offset, y_offset = x_scale / 100, y_scale / 100

    for j, txt in enumerate(list(range(len(target_points)))):
        if j in point_inds:
            ax[0,i].annotate(txt, (target_points[j,2]+x_offset, target_points[j,1]+y_offset), fontsize=3, color = 'white')
            ax[0,i].annotate(txt, (predictions_points[j,2]+x_offset, predictions_points[j,1]+y_offset), fontsize=3, color = 'black')


def retransform_points(points):
    image_shape = [10, 325, 450]
    scale = 10
    zimmer_fluroscence_um_per_pixel_xy: float = 0.325
    points[:,2] = points[:,2] / scale * image_shape[2] + image_shape[2]/2
    points[:,1] = points[:,1] / scale * image_shape[2] + image_shape[1]/2
    points[:,0] = points[:,0] / scale * zimmer_fluroscence_um_per_pixel_xy * image_shape[2] * 3 + image_shape[0]/2 * 3
    return points




# def plot_usable_points(inputs, target_points, run_name):
#     plt.figure(figsize=(20,16),dpi=200)
#     n_test_samples = 459
#     unusable_points = torch.tensor([101, 143, 108, 144, 180, 121, 114, 111, 119, 146, 184, 122, 104, 285, 113, 115, 182, 286, 138, 160, 158, 138, 113, 212, 145, 127, 143, 169, 196, 143, 156, 143, 118, 221, 228, 136, 239, 401, 181, 152, 163, 117, 396, 291, 200, 381, 145, 288, 155, 125, 306, 333, 128, 115, 153, 186, 163, 124, 186, 210,  90, 369, 210, 316, 200, 209, 112, 258, 133, 299, 319, 116, 111, 170, 140, 111, 161, 175, 210, 340, 148, 395, 137, 177, 156, 202, 169, 141, 118, 161, 137, 155, 135, 233, 282, 161, 323, 372, 257, 157, 372, 369, 109, 136, 300, 147, 215, 348, 338])
#     color_uup = (n_test_samples - unusable_points) / n_test_samples

#     print(unusable_points.shape)
#     channeled_image_xy = torch.squeeze(torch.mean(inputs[0], dim = 0), dim = 0)
#     plt.imshow(channeled_image_xy, cmap='bone')
#     target_points = retransform_points(target_points)
#     plt.scatter(target_points[:,2], target_points[:,1], c=color_uup, cmap='winter', s=18)
#     point_inds = list(range(target_points.shape[0]))

#     x_scale = torch.max(target_points[:,2]) - torch.min(target_points[:,2])
#     y_scale = torch.max(target_points[:,1]) - torch.min(target_points[:,1])
#     x_offset, y_offset = x_scale / 100, y_scale / 100

#     for i in point_inds:
#         plt.annotate(str(i), (target_points[i,2]+x_offset, target_points[i,1]+y_offset), fontsize=5, color = "white")

#     plt.colorbar(shrink=0.23)
    
#     plt.savefig('runs/{}/useful_points.png'.format(run_name))



def plot_usable_points(inputs, target_points, run_name):
    plt.figure(figsize=(20,16),dpi=200)
    confused_points = [(104, 108), (42, 37), (25, 49), (43, 63), (61, 100), (51, 50), (65, 28), (45, 100), (47, 43), (107, 97)]

    channeled_image_xy = torch.squeeze(torch.mean(inputs[0], dim = 0), dim = 0)
    plt.imshow(channeled_image_xy, cmap='bone')
    target_points = retransform_points(target_points)
    plt.scatter(target_points[:,2], target_points[:,1], c='white', s=10, alpha = 0.5)
    
    cmap = plt.cm.hsv
    norm = colors.Normalize(vmin=0, vmax=len(confused_points))
    
    x_scale = torch.max(target_points[:,2]) - torch.min(target_points[:,2])
    y_scale = torch.max(target_points[:,1]) - torch.min(target_points[:,1])
    x_offset, y_offset = x_scale / 100, y_scale / 100

    for i, cp in enumerate(confused_points):
        plt.scatter(target_points[cp[0],2], target_points[cp[0],1], color=cmap(norm(i)), s=18, alpha=0.5)
        plt.scatter(target_points[cp[1],2], target_points[cp[1],1], color=cmap(norm(i)), s=18, alpha=0.5)
        x_pos = [target_points[cp[0],2], target_points[cp[1],2]]
        y_pos = [target_points[cp[0],1], target_points[cp[1],1]]
        plt.plot(x_pos, y_pos, color=cmap(norm(i)), linewidth=0.75)
        plt.annotate(str(cp[0]), (target_points[cp[0],2]+x_offset, target_points[cp[0],1]+y_offset), fontsize=5, color=cmap(norm(i)))
        plt.annotate(str(cp[1]), (target_points[cp[1],2]+x_offset, target_points[cp[1],1]+y_offset), fontsize=5, color=cmap(norm(i)))
    
    plt.savefig('runs/{}/useful_points.png'.format(run_name))



def plot_confusion_matrix(assoc, conf_mat, batch_idx, batch_size, run_name):
    n_neurons = assoc.shape[1]    
    conf_new = torch.zeros((n_neurons, n_neurons))
    for a in assoc:
        conf_new[list(range(n_neurons)), a] += 1
    
    if conf_mat == None:
        conf_mat = conf_new
    else:
        conf_mat += conf_new
    
    im = plt.matshow(conf_mat/(batch_size*(batch_idx+1)), norm=LogNorm(), cmap = 'inferno')
    plt.colorbar(im)
    plt.savefig('runs/{}/confusion_matrix.png'.format(run_name))
    # print(assoc.shape)

    torch.save(conf_mat, f'runs/{run_name}/conf_mat.pt')
    return conf_mat