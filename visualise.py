from turtle import width
import torch
import matplotlib.pyplot as plt


def visualise_neurons(inputs, targets, predictions, batch, mode):
    gs_kw = dict(height_ratios=[10.3, 1])
    fig, ax = plt.subplots(2, 2, figsize=(20, 10), dpi=300, gridspec_kw=gs_kw)
    fig.suptitle('batch' + str(batch), fontsize=20)

    for i in range(2):
        target_points = targets[i].reshape(109, 3).cpu()
        predictions_points = predictions[i].reshape(109, 3).cpu()

        plot_scatters(ax, i, inputs.cpu(), target_points, predictions_points)

    plt.savefig('progress_images/' + mode + '_' + str(batch))
    # plt.show()

def plot_scatters(ax, i, inputs, target_points, predictions_points):
    channeled_image_xy = torch.squeeze(torch.mean(inputs[i], dim = 0), dim = 0)
    ax[0,i].imshow(channeled_image_xy, cmap='inferno')

    point_inds = list(range(target_points.shape[0]))
    mask = list(set(torch.where(torch.tensor(target_points) == torch.tensor([0, 0, 0]))[0].numpy()))
    # print("invisible points:", len(mask))
    point_inds = list(filter(lambda x:x not in mask, point_inds))

    target_points = retransform_points(target_points)
    predictions_points = retransform_points(predictions_points)
    
    ax[0,i].scatter(target_points[point_inds,2], target_points[point_inds,1], c="white", s=1.1)
    ax[0,i].scatter(predictions_points[point_inds,2], predictions_points[point_inds,1], c="black", s=1.5)
    # ax[0,i].scatter(torch.mean(target_points[:,2]), torch.mean(target_points[:,1]), c="blue", s=1.1)

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


# def retransform_points(points):
#     image_shape = (21, 650, 900)
#     scale = 10
#     points[:,2] = points[:,2] / scale * image_shape[2] + image_shape[2]/2
#     points[:,1] = points[:,1] / scale * image_shape[2] + image_shape[1]/2
#     points[:,0] = points[:,0] / scale * zimmer_fluroscence_um_per_pixel_xy * image_shape[2] * 3 + image_shape[0]/2 * 3
#     return points

def retransform_points(points):
    image_shape = (21, 650, 900)
    zimmer_fluroscence_um_per_pixel_xy: float = 0.325
    points[:,2] = points[:,2] + image_shape[2]/2
    points[:,1] = points[:,1] + image_shape[1]/2
    points[:,0] = (points[:,0] + image_shape[0]/2) * 3
    return points
