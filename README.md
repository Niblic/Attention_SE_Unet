# Attention_SE_Unet
Attention SE UNet for Tensorflow v2.16 
## Overview

Attention SE U-Net is a deep learning architecture designed for image segmentation tasks. By integrating Squeeze-and-Excitation (SE) blocks and Attention mechanisms, the network enhances feature recalibration and focuses on important regions in the input images. This implementation is built using TensorFlow 2.16.

The primary goal of this project is to provide a high-performance model for segmentation tasks, such as medical imaging or object delineation, while maintaining flexibility for custom datasets.
## Features

    Attention Mechanism: Focuses on relevant image regions, improving segmentation accuracy.
    Squeeze-and-Excitation (SE) Blocks: Dynamically adjusts channel-wise feature importance.
    Fully Convolutional Architecture: Suitable for input images of varying sizes.
    Binary Segmentation Support: Designed for binary mask generation.

## SE-U-Net Architektur:

<table>
  <thead>
    <tr>
      <th>Level</th>
      <th>Heigth/Width</th>
      <th>Channels</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Input Image (RGB)</td>
      <td>512 x 512</td>
      <td>3</td>
    </tr>
    <tr>
      <td>Encoder Block 1</td>
      <td>512 x 512</td>
      <td>32</td>
    </tr>
    <tr>
      <td>Max-Pooling 1</td>
      <td>256 x 256</td>
      <td>32</td>
    </tr>
    <tr>
      <td>Encoder Block 2</td>
      <td>256 x 256</td>
      <td>64</td>
    </tr>
    <tr>
      <td>Max-Pooling 2</td>
      <td>128 x 128</td>
      <td>64</td>
    </tr>
    <tr>
      <td>Encoder Block 3</td>
      <td>128 x 128</td>
      <td>128</td>
    </tr>
    <tr>
      <td>Max-Pooling 3</td>
      <td>64 x 64</td>
      <td>128</td>
    </tr>
    <tr>
      <td>Encoder Block 4</td>
      <td>64 x 64</td>
      <td>256</td>
    </tr>
    <tr>
      <td>Max-Pooling 4</td>
      <td>32 x 32</td>
      <td>256</td>
    </tr>
    <tr>
      <td>Bottleneck Layer</td>
      <td>16 x 16</td>
      <td>512</td>
    </tr>
    <tr>
      <td>Decoder Block 1</td>
      <td>32 x 32</td>
      <td>256</td>
    </tr>
    <tr>
      <td>Decoder Block 2</td>
      <td>64 x 64</td>
      <td>128</td>
    </tr>
    <tr>
      <td>Decoder Block 3</td>
      <td>128 x 128</td>
      <td>64</td>
    </tr>
    <tr>
      <td>Decoder Block 4</td>
      <td>256 x 256</td>
      <td>32</td>
    </tr>
    <tr>
      <td>Output Level</td>
      <td>512 x 512</td>
      <td>1 (Binärsegmentation)</td>
    </tr>
  </tbody>
</table>

## Installation

Clone the Repository:

    git clone https://github.com/Niblic/Attention_SE_Unet.git
    cd Attention_SE_Unet

Install Dependencies: Ensure you have Python 3.8+ installed. Install the required Python packages using pip:

    pip install tensorflow numpy matplotlib pillow

If you use a GPU for training, install the GPU-enabled version of TensorFlow:

    pip install tensorflow-metal  # For macOS with Metal API
    pip install tensorflow-gpu    # For other platforms with NVIDIA GPUs

## Creating Training Data

To train the model, provide images with corresponding masks:

    Images: The input images for segmentation.
    Masks: Binary masks indicating the segmentation region.
    Naming Convention: For each image named image.tiff, provide a mask named image_MASK.tiff.

## Example Code:

load_image_data and load_training_data functions handle image resizing, normalization, and pairing:
  Organize your training data in a directory. For each image (image.tiff), provide a corresponding mask (image_MASK.tiff). Ensure they are stored in the same folder.
  To compile the model we can use different optimsers and loss functions.
  We have Dice and Jacard. Below a sample with Adam and Dice loss function

    # Compile the model with optimizer, loss, and metrics
    optimizer = tf.keras.optimizers.legacy.Adam()
    loss = dice_coef_loss
    metrics = [dice_coef]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


We can start the training with model.fit() using a callback to write some login and save the models.
If need will be this can be removed to save disk space.

    # Training Model code sample
    def train_model(model, input, output, current_epoch):
        callbacks = [
            #keras.callbacks.EarlyStopping(monitor="loss", patience=4),
            keras.callbacks.TensorBoard(
                log_dir=log_directory,
                update_freq="epoch",
                write_graph=True,
                write_images=True,
                write_steps_per_second=True
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=model_directory + "/checkpoint-{epoch}",
                save_freq="epoch",
            ),
            keras.callbacks.ProgbarLogger(),
        ]
        # Train the model with input and output data
        # with 128x128 we can manage a batch of 100 and more
        # with 512x512 we get it working with 25 maybe more M4 needs 13 Sec
        #model.fit(input, output, initial_epoch=current_epoch, batch_size=len(input), epochs=1500, callbacks=callbacks)
        model.fit(input, output, initial_epoch=current_epoch, batch_size=25, epochs=1500, callbacks=callbacks)

To install GPU support for M4 processors follow this instructions:

    https://medium.com/bluetuple-ai/how-to-enable-gpu-support-for-tensorflow-or-pytorch-on-macos-4aaaad057e74


      Loads and preprocesses individual images or masks (e.g., resizing, normalization).

  load_training_data:
      Scans the specified directory.
      Matches images with masks using the naming convention.
      Creates and returns the input and output arrays for training.

These functions ensure that the training data is prepared efficiently and accurately for use in the network.

    def load_image_data(path, mask=False, target_size=(512, 512)):
        # Select Color schema
        color_mode = "grayscale" if mask else "rgb"
        # Transform image into required target_size
        image = keras.utils.load_img(path, color_mode=color_mode, target_size=target_size)
        # Change the image into an array required for the network to work
        image_data = keras.utils.img_to_array(image)
        # NCHW -> NHWC might need the channel last.. that is required for the network to work
        image_data = np.moveaxis(image_data, 0, -1) if image_data.shape[0] < image_data.shape[-1] else image_data
        # Normalize the mask to 0 and 1
        return tf.round(image_data / 255.0) if mask else image_data / 255.0
    
    def load_training_data(target_size=(512, 512)):
        inputs = []
        outputs = []
        for entry in os.scandir(data_directory):
            if entry.is_file and entry.name.endswith("_MASK.tiff"):
                # Mask  will be added
                outputs.append(load_image_data(entry.path, mask=True, target_size=target_size))
                # Input will be added
                inputs.append(load_image_data(entry.path.replace("_MASK.tiff", ".tiff"), target_size=target_size))
        # Sicherstellen, dass die Daten NHWC sind und als Float32 gespeichert werden
        return np.array(inputs, dtype=np.float32), np.array(outputs, dtype=np.float32)

## Results

During training, the model outputs loss metrics and Dice coefficient values. Sample results are saved in the results directory.

    Input Image:

    Ground Truth Mask:

    Model Prediction:

## References

    Original U-Net Paper: U-Net: Convolutional Networks for Biomedical Image Segmentation
    SE-Net Paper: Squeeze-and-Excitation Networks    
