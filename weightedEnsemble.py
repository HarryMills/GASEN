def weighted_ensemble(weights, models, inputs):
    # Assigning empty array to store 2D array of model predictions
    predictions = []

    # Loop through all models
    for i in range(len(models)):
        # Make predictions for each model
        predictions.append(models[i].predict(inputs))
        # Multiply prediction by corresponding weight
        predictions[i] = [x * weights[i] for x in predictions[i]]

    # Sum of weighted predictions
    predictionsSum = [sum(x) for x in zip(*predictions)]

    return predictionsSum