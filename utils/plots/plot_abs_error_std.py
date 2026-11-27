import numpy as np
import matplotlib.pyplot as plt
import os
from utils.torch_funcs.function_cuda_to_cpu import cuda_to_cpu
from utils.plots.function_robust_data_range import compute_robust_data_range
import matplotlib.colors as colors
from utils.plots.function_fields_plot_prep import fields_plot_prep

def plot_abs_error_std(true_field, posterior_mean, standard_deviation, triangulation, suptitle=None, results_path=None, r_error_instead_std=False):
    """
    Create 4 subplots in a 2 by 2 grid fashion.

    Parameters:
        true_field (np.ndarray): The true, noisy field data as a 2D numpy array.
        posterior_mean (np.ndarray): The posterior mean field data as a 2D numpy array.
        standard_deviation (float): The standard deviation value for dividing the error in the fourth subplot.
        mesh (np.ndarray): The 2D numpy array representing the mesh grid for the data.
    """
        # flatten if still not flattened
    true_field = fields_plot_prep(true_field)
    posterior_mean = fields_plot_prep(posterior_mean)
    standard_deviation = fields_plot_prep(standard_deviation)

    # This flag determines if we plot the triangle constant values or the node values
    if len(triangulation.triangles) == posterior_mean.size(0):
        flag_tri_const = True
    elif len(triangulation.x) == posterior_mean.size(0):
        flag_tri_const = False
    else:
        raise ValueError("The triangulation does not match the field size.")
    
    # Convert the data to CPU
    true_field, posterior_mean, standard_deviation = cuda_to_cpu(true_field, posterior_mean, standard_deviation)

    # Calculate the absolute error between the true_field and posterior_mean
    absolute_error = np.abs(posterior_mean - true_field)

    # Create a figure and a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Plot the true, noisy field in the first subplot
    # try:
    #     im1 = axes[0, 0].pcolormesh(mesh[0].numpy(), mesh[1].numpy(), true_field)
    # except:
    #     im1 = axes[0, 0].pcolormesh(true_field)
    if flag_tri_const:
        im1 =  axes[0, 0].tripcolor(triangulation, facecolors=true_field)
    else:
        im1 =  axes[0, 0].tricontourf(triangulation, true_field)
    axes[0, 0].set_title('True, Noisy Field')

    # Plot the posterior mean field in the second subplot
    # try:
    #     im2 = axes[0, 1].pcolormesh(mesh[0].numpy(), mesh[1].numpy(), posterior_mean)
    # except:
    #     im2 = axes[0, 1].pcolormesh(posterior_mean)
    if flag_tri_const:
        im2 =  axes[0, 1].tripcolor(triangulation, facecolors=posterior_mean)
    else:
        im2 =  axes[0, 1].tricontourf(triangulation, posterior_mean)
    axes[0, 1].set_title('Posterior Mean Field')

    # Plot the absolute error in the third subplot
    # try:
    #     im3 = axes[1, 0].pcolormesh(mesh[0].numpy(), mesh[1].numpy(), absolute_error)
    # except:
    #     im3 = axes[1, 0].pcolormesh(absolute_error)
    if flag_tri_const:
        im3 =  axes[1, 0].tripcolor(triangulation, facecolors=absolute_error)
    else:
        im3 =  axes[1, 0].tricontourf(triangulation, absolute_error)
    axes[1, 0].set_title('Absolute Error')

    # Plot the error divided by the standard deviation in the fourth subplot
    if r_error_instead_std:
        error_divided_by_std = absolute_error / true_field
        error_divided_by_std = np.nan_to_num(error_divided_by_std, nan=0.0, posinf=0.0, neginf=0.0)
        # only plot relevant range and dont care for individual outliers
        low_value, high_value = compute_robust_data_range(error_divided_by_std)
        norm = colors.PowerNorm(gamma=1, vmin=low_value, vmax=high_value)
        # try:
        #     im4 = axes[1, 1].pcolormesh(mesh[0].numpy(), mesh[1].numpy(), error_divided_by_std, norm=norm)
        # except:
        #     im4 = axes[1, 1].pcolormesh(error_divided_by_std, norm=norm)
        if flag_tri_const:
            im4 =  axes[1, 1].tripcolor(triangulation, facecolors=error_divided_by_std, norm=norm)
        else:
            im4 =  axes[1, 1].tricontourf(triangulation, error_divided_by_std, norm=norm)
    else:
        error_divided_by_std = absolute_error / standard_deviation
        # try:
        #     im4 = axes[1, 1].pcolormesh(mesh[0].numpy(), mesh[1].numpy(), error_divided_by_std)
        # except:
        #     im4 = axes[1, 1].pcolormesh(error_divided_by_std)
        if flag_tri_const:
            im4 =  axes[1, 1].tripcolor(triangulation, facecolors=error_divided_by_std)
        else:
            im4 =  axes[1, 1].tricontourf(triangulation, error_divided_by_std)

    if r_error_instead_std:
        axes[1, 1].set_title('Relative Error')
    else:
        axes[1, 1].set_title('Error / Standard Deviation')

    # Add colorbars to the subplots
    fig.colorbar(im1, ax=axes[0, 0])
    fig.colorbar(im2, ax=axes[0, 1])
    fig.colorbar(im3, ax=axes[1, 0])
    fig.colorbar(im4, ax=axes[1, 1])

    # main title
    plt.suptitle(suptitle)

    # Adjust layout to prevent overlapping of titles and labels
    plt.tight_layout()

    if results_path is not None:
        plt.savefig(os.path.join(results_path, suptitle + ".png"))

    # Display the plot
    plt.show()
    plt.close()

# Example usage:
# Assume true_field, posterior_mean, standard_deviation, and mesh are already defined numpy arrays.
# plot_subplots(true_field, posterior_mean, standard_deviation, mesh)
