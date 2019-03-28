def ensemble_fitness(weights, models, inputs, targets, value):
    import numpy as np
    import sklearn
    from sklearn import linear_model

    weights = weights/sum(weights)

    # Loop through all networks
    for i in range(len(models)):
        print(models[i])
        # predictions = models[i].predict(inputs)

    # Weighted prediction for each network

    # Sum of weighted predictions

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
