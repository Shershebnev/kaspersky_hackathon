import os
import time
import warnings
import zipfile

from keras.callbacks import ModelCheckpoint
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.utils.data_utils import get_file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

def parse_test_data(path):
    """Splits test csv files by tag and saves into separate directories
    based on filename

    Args:
        path: path to the directory with files

    Returns:
        list of test filenames without file extension
    """
    filetags = []
    for filename in [item for item in os.listdir(path) if item.endswith(".csv")]:
        print("Parsing {}".format(filename))
        filetag = os.path.splitext(filename)[0]
        filetags.append(filetag)
        tagdir = os.path.join(path, filetag)
        if not os.path.isdir(tagdir):
            os.mkdir(tagdir)
        data = pd.read_csv(os.path.join(path, filename))
        data = data.dropna(axis = 0, how = "any")  # removing empty rows
        columns = data.columns[1:]  # removing time column
        for column in columns:
            column_fname = "{tag}_{column}.csv".format(tag = filetag, column = column)
            fpath = os.path.join(tagdir, column_fname)
            if not os.path.isfile(fpath):
                f = open(os.path.join(tagdir, column_fname), "w")
                col_data = data[[column]].values
                col_data = np.reshape(col_data, (col_data.shape[0], )).tolist()
                f.write("{}\n".format("\n".join(map(str, col_data))))
                f.close()
    return filetags

def parse_train_data(path):
    """Splits training set file into separate files, one for each column
    except time

    Args:
        path: path to training set file
    """
    data = pd.read_csv(path)
    columns = data.columns[1:]
    for column in columns:
        f_path = "{}.csv".format(column)
        f = open(f_path, "w")
        val = data[[column]].values
        a = np.reshape(val, (val.shape[0], )).tolist()
        f.write("{}\n".format("\n".join(map(str, a))))
        f.close()

def load_data(filename, seq_len, scaler, validation_split = True):
    """Loads data from the file and converts it into batches. Files assumed
    to have only one column with data

    Args:
        filename: path to the file
        seq_len: length of one sequence in the batch
        scaler: scaler object, like MinMaxScaler from scikit-learn. Should
            have fit_transform method.
        validation_split: whether to split part of the data into validation
            set. Supposed to be True when loading training set and False for
            test set. Split ratio is 90:10

    Returns:
        list of numpy arrays. If validation_split == True, returns
        [x_train, y_train, x_validation, y_validation].
        If validation_split == False, returns [x_test, y_test]
    """
    f = open(filename)
    data = list(map(lambda x: float(x.strip()), f.readlines()))
    f.close()
    sequence_length = seq_len + 1
    batches = []
    for idx in range(len(data) - sequence_length):
        batches.append(data[idx:idx + sequence_length])
    batches = scaler.fit_transform(batches)
    batches = np.array(batches)

    if validation_split:
        idx = round(0.9 * batches.shape[0])

        training_set = batches[:idx, :]
        x_train = training_set[:, :-1]
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        y_train = training_set[:, -1]

        validation_set = batches[idx:, :]
        x_validation = validation_set[:, :-1]
        x_validation = np.reshape(x_validation, (x_validation.shape[0], 
                                                 x_validation.shape[1], 1))
        y_validation = validation_set[:, -1]
        return [x_train, y_train, x_validation, y_validation]
    else:
        x_test = batches[:, :-1]
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        y_test = batches[:, -1]
        return [x_test, y_test]

def build_model(layers_sizes, dropout = 0.2, activation = "linear", loss = "mse",
                optimizer = "rmsprop"):
    """Creates NN model.

    Args:
        layers_sizes: list of layers sizes: 
            input layer size size, first LSTM layer size, second LSTM layer 
            size, output layer size
        dropout: dropout probability. DEFAULT: 0.2
        activation: activation function. DEFAULT: linear
        loss: how loss is calculated. DEFAULT: mse
        optimizer: optimizer used. DEFAULT: rmsprop

    Returns:
        compiled model
    """
    model = Sequential()

    # first LSTM
    model.add(LSTM(input_dim = layers_sizes[0], output_dim = layers_sizes[1], 
              return_sequences = True))
    model.add(Dropout(dropout))

    # second LSTM
    model.add(LSTM(layers_sizes[2], return_sequences = False))
    model.add(Dropout(dropout))

    # fully connected output layer
    model.add(Dense(output_dim = layers_sizes[3]))
    model.add(Activation(activation))

    model.compile(loss = loss, optimizer = optimizer)

    return model

def predict(model, data):
    """Predicts data point by point

    Args:
        model: fitted model
        data: test data
    """
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.shape[0], ))
    return predicted

def plot_results(predicted, true_data, filepath):
    """Creates plot with two figures: for predicted and true data

    Args:
        predicted: predicted data, as returned by predict()
        true_data: original target data
        filepath: where to save file, should have .png extension
    """
    fig, ((ax1, ax2)) = plt.subplots(nrows = 2, ncols = 1)
    ax1.plot(predicted)
    ax1.set_title("Predicted")
    ax2.plot(true_data)
    ax2.set_title("True data")
    plt.tight_layout()
    plt.savefig(filepath, dpi = 400)
    plt.close()

def zip_data(path, zip_handler):
    for root, dirs, files in os.walk(path):
        for filename in files:
            zip_handler.write(os.path.join(root, filename))

def convert_time(seconds):
    """Converts time in seconds into days, hours, minutes and seconds

    Args:
        seconds: number of seconds

    Returns:
        seconds converted into the number of days, hours, minutes and seconds
    """
    seconds = int(seconds)  # removing ms
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    return days, hours, minutes, seconds

def select_best_model(path, pattern, compare_fn):
    """Selects best model from checkpoints

    Args:
        path: path to directory with checkpoints
        pattern: regexp compiled pattern for checkpoint name matching
        compare_fn: function for checkpoints comparison. Should take two
            arguments and return 1 if first is better. Args assumed to be floats

    Returns:
        path to best checkpoint
    """
    best_value = None
    best_checkpoint = ""
    for checkpoint in [item for item in os.listdir(path) if item.endswith(".h5")]:
        value = float(pattern.findall(checkpoint)[0])
        if best_value is None:
            best_value = value
            best_checkpoint = checkpoint
        else:
            cmp_val = compare_fn(value, best_value)
            if cmp_val == 1:
                best_value = value
                best_checkpoint = checkpoint
    return opj(path, best_checkpoint)

global_start = time.time()
train_file = "train.csv"
if not os.path.isfile(train_file):
    print("Downloading data")
    path = get_file("./kasper.zip", origin = "https://events.kaspersky.com/hackathon/uploads/kaspersky_hackathon_1.zip")
    z = zipfile.ZipFile(path, "r")
    z.extractall(".")
    z.close()
test_dir = "./test/"
parse_train_data(train_file)

# hyperparameters
epochs = 10
seq_len = 50
lstm_first = 50
lstm_second = 100
batch_size = 512

tag_label = "tag00"
checkpoints_dir = "./checkpoints"
predictions_dir = "./predictions"
if not os.path.isdir(checkpoints_dir):
    os.mkdir(checkpoints_dir)
checkpoint_filepath = os.path.join(checkpoints_dir,
    tag_label + "_epoch_{epoch:02d}_val_loss_{val_loss:.5f}.h5")
checkpoint = ModelCheckpoint(checkpoint_filepath, monitor = "val_loss",
                            verbose = 1, save_best_only = True, mode = "min")
callbacks_list = [checkpoint]
scaler = MinMaxScaler(feature_range = (0, 1), copy = True)

print("Loading training data")
x_train, y_train, x_validation, y_validation = load_data("{}.csv".format(tag_label),
                                                         seq_len, scaler, True)

print("Fitting model")
model = build_model([1, lstm_first, lstm_second, 1])
model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs,
          validation_split = 0.1, callbacks = callbacks_list, verbose = 2)

print("Loading best model")
pattern = re.compile("epoch_\d\d_val_loss_(\d+\.\d+).h5")
compare_fn = lambda x, y: 1 if x < y else 0
best_chkp = select_best_model(checkpoints_dir, pattern, compare_fn)
model = load_model(best_chkp)

print("Testing using validation set")
predicted = predict(model, x_validation)
plot_results(predicted, y_validation, "{}_validation.png".format(tag_label))

print("Loading test data and making predictions")
if not os.path.isdir(predictions_dir):
    os.mkdir(predictions_dir)
filetags = parse_test_data(test_dir)
for filetag in sorted(filetags, key = lambda x: int(x[:2])):
    start = time.time()
    print("Predicting using test data from {}".format(filetag))
    test_data_path = os.path.join(test_dir, filetag, "{}_{}.csv".format(filetag, tag_label))
    x_test, y_test = load_data(test_data_path, seq_len, scaler, False)
    predicted = predict(model, x_test)
    pred_file = os.path.join(predictions_dir, "{}_{}.txt".format(filetag, tag_label))
    f = open(pred_file, "w")
    f.write("{}\n".format("\n".join(map(str, predicted.tolist()))))
    f.close()
    plot_path = os.path.join(predictions_dir, "{}_{}.png".format(filetag, tag_label))
    plot_results(predicted, y_test, plot_path)
    print("Predicted using test data from {} in {:d} days {:d} hours {:d} minutes {:d} seconds".format(filetag,
                                            *convert_time(time.time() - start)))

print("Creating archives")
checkpoints_zip = zipfile.ZipFile("{}_checkpoints.zip".format(tag_label), "w",
                                  zipfile.ZIP_DEFLATED)
zip_data(checkpoints_dir, checkpoints_zip)
checkpoints_zip.close()
predictions_zip = zipfile.ZipFile("{}_predictions.zip".format(tag_label), "w",
                                  zipfile.ZIP_DEFLATED)
zip_data(predictions_dir, predictions_zip)
predictions_zip.close()

print("DONE in {:d} days {:d} hours {:d} minutes {:d} seconds".format(*convert_time(time.time() - global_start)))



