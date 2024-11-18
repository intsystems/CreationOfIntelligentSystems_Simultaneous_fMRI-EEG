import matplotlib.pyplot as plt


def show_slices(slices):
    _, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
        axes[i].axis('off')
    plt.show()