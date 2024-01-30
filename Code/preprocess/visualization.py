def visualize_image(axes_array, images_array, labels_array, figure):
    axes_array = axes_array.flatten()

    # assert len(axes_array) == len(images_array) == len(labels_array)
    assert len(axes_array) <= len(images_array)
    assert len(axes_array) <= len(labels_array)

    for i, ax in enumerate(axes_array):
        img = ax.imshow(images_array[i], cmap="gray")
        figure.colorbar(img, ax=ax)
        ax.set_title(labels_array[i])
        ax.axis("off")
