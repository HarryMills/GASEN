def ensemble_fitness(weights, models, inputs, targets, value):
    import numpy as np
    import sklearn
    from sklearn import linear_model

    weights = [x / sum(weights) for x in weights]

    predictions = []

    # Loop through all networks
    for i in range(len(models)):
        predictions.append(models[i].predict(inputs))
        predictions[i] = [x * weights[i] for x in predictions[i]]

    # Sum of weighted predictions
    predictionsFull = [sum(x) for x in zip(*predictions)]

    # Calculate fitness values
    bias = (np.mean(predictions)-np.mean(targets))**2
    variance = np.var(predictions-targets)
    error = bias+variance
    # mse =

    # Setting output fitness value
    if value == "mse":
        ensembleFit = mse
    elif value == "bias":
        ensembleFit = bias
    elif value == "variance":
        ensembleFit = variance
    elif value == "error":
        ensembleFit = error
    else:
        ensembleFit = mse

    # Returning fitness value
    return ensembleFit
