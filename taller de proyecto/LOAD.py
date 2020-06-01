import numpy as np

def load_multigpu_checkpoint_weights(model, h5py_file):
    """
    Loads the weights of a weight checkpoint from a multi-gpu
    keras model.

    Input:

        model - keras model to load weights into

        h5py_file - path to the h5py weights file

    Output:
        None
    """

    print("Setting weights...")
    with h5py.File(h5py_file, "r") as file:

        # Get model subset in file - other layers are empty
        weight_file = file["model_1"]

        for layer in model.layers:

            try:
                layer_weights = weight_file[layer.name]

            except:
                # No weights saved for layer
                continue

            try:
                weights = []
                # Extract weights
                for term in layer_weights:
                    if isinstance(layer_weights[term], h5py.Dataset):
                        # Convert weights to numpy array and prepend to list
                        weights.insert(0, np.array(layer_weights[term]))

                # Load weights to model
                layer.set_weights(weights)

            except Exception as e:
                print("Error: Could not load weights for layer:", layer.name, file=stderr)