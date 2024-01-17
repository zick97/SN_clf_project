from python_scripts.utils import *
import IPython

# Function to automate the training process and plotting
def trainPipeline(X_train, y_train, X_val, y_val, classes=6, want_callback=True):
    # Create the model with the same architecture as before
    model = createModel(input_shape=X_train.shape[1], classes=classes)

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy',
                optimizer=keras.optimizers.Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.999),
                metrics=['accuracy'])
    
    if classes == 2:
        model.compile(loss='binary_crossentropy',
                    optimizer=keras.optimizers.Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.999),
                    metrics=['accuracy'])

    import time
    start = time.time()

    # Create an instance of the custom callback
    if want_callback:
        precision_recall_callback = PrecisionRecallCallback(validation_data=(X_val, y_val))
        # Fit the model using the training data and validation data
        history = model.fit(X_train, y_train, epochs=100, steps_per_epoch=64, 
                            validation_data=(X_val, y_val),
                            callbacks=[keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
                                    precision_recall_callback])
    else:
        history = model.fit(X_train, y_train, epochs=100, steps_per_epoch=64, 
                            validation_data=(X_val, y_val),
                            callbacks=[keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)])
    
    # Clear the printed output
    IPython.display.clear_output()

    # Compute the time taken to train the model
    end = time.time()
    time = end - start

    # Plot the learning curves with all the metrics
    if want_callback: plotHistory(history, precision_recall_callback)
    else: plotHistory(history)

    return model, time