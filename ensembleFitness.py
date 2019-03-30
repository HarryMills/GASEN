def ensemble_fitness(weights, models, inputs, targets, value):
    import numpy as np
    import sklearn
    from sklearn import linear_model
    from sklearn import metrics

    # Calculate weights in correct format
    weights = [x / sum(weights) for x in weights]

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

    # Calculate predictions metrics
    bias = (np.mean(predictionsSum)-np.mean(targets))**2
    variance = np.var(predictionsSum-targets)
    error = bias+variance
    mse = metrics.mean_squared_error(predictionsSum, targets)
    mae = metrics.mean_absolute_error(predictionsSum, targets)

    # Setting output fitness value
    if value == "mse":
        ensembleFit = mse
    elif value == "mae":
        ensembleFit = mae
    elif value == "bias":
        ensembleFit = bias
    elif value == "variance":
        ensembleFit = variance
    elif value == "error":
        ensembleFit = error
    else:
        # If error with input then set it to mse as default
        ensembleFit = mse

    # Returning fitness value to minimise
    return ensembleFit
