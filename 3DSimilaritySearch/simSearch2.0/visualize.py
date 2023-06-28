import numpy as np
from matplotlib import pyplot as plt

def visualizeResults(inputObject, mostSimilarObjects, numObjectsToReturn):
    
    rows = int(np.ceil(np.sqrt(numObjectsToReturn + 1)))
    cols = int(np.ceil((numObjectsToReturn + 1) / rows))

    fig = plt.figure(figsize=(cols * 7, rows * 7))

    ax = fig.add_subplot(rows, cols, 1, projection='3d')
    ax.scatter(inputObject[0, :, 0], inputObject[0, :, 1], inputObject[0, :, 2])
    ax.title.set_text('Input object')

    for i, similarObject in enumerate(mostSimilarObjects):
        ax = fig.add_subplot(rows, cols, i + 2, projection='3d')
        ax.scatter(similarObject[:, 0], similarObject[:, 1], similarObject[:, 2])
        ax.title.set_text('Similar object {}'.format(i + 1))

    plt.subplots_adjust(wspace=0.1, hspace=0.4)
    plt.show()