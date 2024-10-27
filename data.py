import numpy as np

from dataset import Dataset


# load dataset
def load_data():
    path = r"lines_dataset2"
    data = Dataset.load_from_file(path).data

    x_data, y_data = zip(*data)
    x_data = np.array(x_data, dtype="uint8")  # Convert x_data to a NumPy array
    y_data = np.array(y_data)  # Convert y_data to a NumPy array

    # Shuffle the data
    indices = np.arange(len(x_data))
    np.random.shuffle(indices)
    x_data = x_data[indices]
    y_data = y_data[indices]

    length = len(x_data)

    x_train, x_test = np.split(x_data, [int(length * 0.9)])
    y_train, y_test = np.split(y_data, [int(length * 0.9)])

    return x_train, y_train, x_test, y_test, None, None
