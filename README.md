# Attention_SE_Unet
Attention SE UNet for Tensorflow v2.16 

TensorFlow 2.x has made some changes that can lead to errors when trying to execute older networks against it. 
In particular, the way TensorFlow 2.x handles symbolic tensors has changed.

More details about SE-Unet can be found under:
https://github.com/hujie-frank/SENet/blob/master/README.md


This UNet will take an input image of size 512x512x(3/1) and create an output mask of dimension 1.
So the output will be a mask of binary values.

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

## Creating Training Data

To train the model effectively, we need a set of images accompanied by their corresponding masks. Each mask represents the target outcome for its associated image.
# Requirements for Training Data

  Images and Masks: Each image must have a matching binary mask (for binary segmentation tasks).

  Naming Convention: To ensure the correct association between images and masks, we use a consistent naming convention. The mask file is named using the corresponding image name with _MASK appended before the file extension (e.g., image.tiff → image_MASK.tiff).

  Directory Structure: Store both the image and its mask in the same directory. This allows us to programmatically pair the images and masks based on their names.

## How It Works

The code processes the directory to:

  Identify and load images and masks based on the naming convention.
  
  Resize or normalize images and masks as needed (e.g., to ensure consistent dimensions).

  Create two arrays:
  
    Input Array: Contains the images.
      
    Output Array: Contains the corresponding masks, aligned with the images.

## Code

We provide two key functions to facilitate the creation of training data:

  load_image_data:
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

    
