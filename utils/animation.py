import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
from os import listdir


def animate_results(path, keyword, ext='.png'):
    """ Creates an animation using all images under the given path.
    Note: To run in jupyter, you will need the folloqing code:
        from IPython.display import HTML
        HTML(animate_results(path, keyword).to_jshtml())
    HTML() doesn't work inside of functions, it has to be called from the cell
    directly.

    Args:
        path (str): path
        keyword (str): only chose image if the filename contains keyword.
        ext (str): only chose image if the file has given extension.

    Returns:
        ArtisticAnimation object, can be run as described above.
    """
    file_names = sorted(listdir(path))
    img_names = [f for f in file_names if keyword in f and ext in f]
    img_list = [mpimg.imread(path + name) for name in img_names]

    plt.rcParams['animation.embed_limit'] = 2 ** 128
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(i, animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000,
                                    blit=True)
    return ani


def show_last_batch(path, keyword, ext='.png'):
    """ Creates an animation using all images under the given path.
    Note: To run in jupyter, you will need the folloqing code:
        from IPython.display import HTML
        HTML(animate_results(path, keyword).to_jshtml())
    HTML() doesn't work inside of functions, it has to be called from the cell
    directly.

    Args:
        path (str): path
        keyword (str): only chose image if the filename contains keyword.
        ext (str): only chose image if the file has given extension.

    Returns:
        ArtisticAnimation object, can be run as described above.
    """
    file_names = sorted(listdir(path))
    img_names = [f for f in file_names if keyword in f and ext in f]
    last_batch_name = img_names[-1]
    last_batch = mpimg.imread(path + last_batch_name)

    plt.figure(figsize=(8, 8))
    plt.imshow(last_batch)
    plt.show()
