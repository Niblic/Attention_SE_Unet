def cnn(input_shape=(512, 512, 3)):
    model = Sequential()
    
    # First Convolutional Layer
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    
    # Second Convolutional Layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    
    # Third Convolutional Layer
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    
    # Flatten the Daten
    model.add(Flatten())
    
    # Dropout Layer für Regularisierung
    model.add(Dropout(0.5))
    
    # Fully Connected Layer
    model.add(Dense(128, activation='relu'))
    
    # Output layer for 2 Classes (Model1 oder Model2)
    model.add(Dense(2, activation='softmax'))  # Softmax, da es zwei Klassen gibt
    
    return model
