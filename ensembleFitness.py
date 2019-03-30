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

    # Calculating bias and variance for use in error if selected
    bias = (np.mean(predictionsSum)-np.mean(targets))**2
    variance = np.var(predictionsSum-targets)

    # Setting output fitness value
    if value == "mse":
        ensembleFit = metrics.mean_squared_error(predictionsSum, targets)
    elif value == "mae":
        ensembleFit = metrics.mean_absolute_error(predictionsSum, targets)
    elif value == "bias":
        ensembleFit = bias
    elif value == "variance":
        ensembleFit = variance
    elif value == "error":
        ensembleFit = bias+variance
    else:
        # If error with input then set it to mse as default
        ensembleFit = metrics.mean_squared_error(predictionsSum, targets)

    # Returning fitness value to minimise
    return ensembleFit
