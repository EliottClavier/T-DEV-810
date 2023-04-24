import tensorflow as tf
import numpy as np
from IPython.display import Image, display
import matplotlib.cm as cm
from keras.utils import load_img, img_to_array, array_to_img

def get_img_array(img_path, size):
    # `img` is a PIL image of size 256x256
    img = load_img(img_path, target_size=size, grayscale=True)
    # `array` is a float32 Numpy array of shape (256, 256, 3)
    array = img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    #grads = tf.expand_dims(grads, axis=0)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def generate_superimposed_gradcam(img_path, heatmap, alpha=0.4):
    # Load the original image
    img = load_img(img_path)
    img = img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = array_to_img(superimposed_img)

    return superimposed_img

def display_heatmaps(img_path, heatmaps, superimposed=False):
    import matplotlib.pyplot as plt

    # Define the number of columns for the subplot grid
    n_cols = min(len(heatmaps), 5)

    # Calculate the number of rows based on the number of columns and total number of images
    n_rows = (len(heatmaps) - 1) // n_cols + 1

    # Create a figure with the desired size
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))

    for i, heatmap in enumerate(heatmaps):
        # Calculate the row and column index for this heatmap
        row = i // n_cols
        col = i % n_cols

        # Get the corresponding subplot
        ax = axes[row, col] if n_rows > 1 else axes[col]

        # Set the title of the subplot to the layer name
        ax.set_title(heatmap['layer_name'])

        if superimposed:
            superimposed_img = generate_superimposed_gradcam(img_path, heatmaps[i]["image"])
            # Plot the heatmap
            ax.imshow(superimposed_img)
        else:
            # Plot the heatmap
            ax.imshow(heatmaps[i]["image"], cmap='jet')

    # Hide the remaining subplots that are not used
    for i in range(len(heatmaps), n_rows * n_cols):
        # Calculate the row and column index for this subplot
        row = i // n_cols
        col = i % n_cols

        # Get the corresponding subplot
        ax = axes[row, col] if n_rows > 1 else axes[col]

        # Hide the subplot
        ax.axis('off')

    # Adjust the spacing between subplots
    fig.tight_layout()

    # Show the plot
    plt.show()
