def load_data():
    import pickle
    import os

    # Initialise empty models array
    models = []

    # Loop through networks directory
    for root, dirs, files, in os.walk("networks"):
        # Loop through all files in directory
        for file in files:
            # Find any files with .pickle extension
            if file.endswith(".pickle"):
                # Open pickle file
                pickle_in = open(("networks/" + file), "rb")
                # Add model to the models array
                models.append(pickle.load(pickle_in))

    # Return the found models in an array
    return models
