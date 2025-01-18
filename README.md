# Attention_SE_Unet
Attention SE UNet for Tensorflow v2.16 

TensorFlow 2.x has made some changes that can lead to this error. 
In particular, the way TensorFlow 2.x handles symbolic tensors has changed.

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


    
    
