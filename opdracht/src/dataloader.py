from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from scipy.io import arff

Tensor = torch.Tensor
Dataframe = pd.DataFrame


def apply_padding(
    df: Dataframe, seq_length: int, offset: int, padding_id: int
) -> Tuple[Tensor, Tensor]:
    print(f"applied padding, sequence length: {len(df)} ==> {seq_length}")
    # create list with the available data points
    time = torch.arange(0, len(df)) + offset
    # create a list with the padding id
    padding = torch.tensor([padding_id] * (seq_length - len(df)))
    # combine both list to one
    time = torch.cat([padding, time]).reshape(1, -1)
    window = torch.arange(0, 1).reshape(-1, 1)
    return time, window


def apply_window(
    df: Dataframe,
    seq_length: int,
    min_seq_length: int,
    continuous_window: bool,
    offset: int,
    padding_id: int,
) -> Tensor:
    if continuous_window:
        # calculate the number of windows
        n_window = int(len(df) - seq_length + 1)
        # if one of more windows are possible
        if n_window > 0:
            time = torch.arange(0, seq_length).reshape(1, -1) + offset
            window = torch.arange(0, n_window).reshape(-1, 1)
        # if the length is more or equal to the minimum number of samples
        elif len(df) >= min_seq_length:
            time, window = apply_padding(df, seq_length, offset, padding_id)
        else:
            print(
                f"chunk is to small to create at least 1 sequence "
                f"even with padding: {len(df)} < {min_seq_length}"
            )
            time = torch.Tensor([])
            window = torch.Tensor([])
    else:
        # calculate the number of windows
        n_window = int(np.floor(len(df) / seq_length))
        # if one of more windows are possible
        if n_window > 0:
            time = torch.arange(0, seq_length).reshape(1, -1) + offset
            window = torch.arange(0, n_window * seq_length,
                                  seq_length).reshape(-1, 1)
        # if the length is more or equal to the minimum number of samples
        elif len(df) >= min_seq_length:
            time, window = apply_padding(df, seq_length, offset, padding_id)
        else:
            print(
                f"chunk is to small to create at least "
                f"1 sequence even with padding: {len(df)} < {min_seq_length}"
            )
            time = torch.Tensor([])
            window = torch.Tensor([])

    idx = time + window
    return idx


class Dataset:
    def __init__(
        self,
        url: str,
        path: str,
        filename: str,
        per_chunk: bool,
        continuous_window: bool,
        seq_length: int,
        min_seq_length: int,
        fractions: list([]),
        random: bool,
    ) -> None:
        # store all input variables
        self.path = path
        self.filename = filename
        self.per_chunk = per_chunk
        self.continuous_window = continuous_window
        self.seq_length = seq_length
        self.min_seq_length = min_seq_length
        self.fractions = fractions
        self.random = random
        self.datapath = Path(path + "/datasets/" + filename)
        # create empty variables
        self.dataset = pd.DataFrame([])
        self.chunk_lengths = []
        self.idx = torch.Tensor()
        self.id_train = []
        self.id_test = []
        self.id_valid = []
        # run the basic functions
        # download the dataset and store on the drive
        self.download_data(url)
        # process the data into sequences
        self.process_data()
        # split the dataset into train/test/validation
        self.split_dataset()

    def download_data(self, url: str) -> None:
        # check if the dataset is already on the drive,
        # if not download the dataset
        if self.datapath.is_file():
            print("data has already been downloaded, file exists")
        else:
            print("download and store dataset to file")
            tf.keras.utils.get_file(
                self.filename, origin=url, untar=False, cache_dir=self.path
            )

    def process_data(self) -> None:
        # load the dataset and create a pandas DataFrame
        data = arff.loadarff(self.datapath)
        df = pd.DataFrame(data[0])
        # Trick to apply padding, create a reference to a
        # row with zero which will be added to the dataframe
        padding_id = len(df)

        if self.per_chunk:
            # if the data is split per chunk, first detect all chunks.
            # each chunk will get an unique number
            df["eyeDetection"] = df["eyeDetection"].astype(int)
            df["chunk_id"] = (
                (df["eyeDetection"] !=
                 df["eyeDetection"].shift()).ne(0).cumsum()
            )
            for chunk in df["chunk_id"].unique():
                # determine the length of the chunk
                self.chunk_lengths.append(sum(df["chunk_id"] == chunk))
                # get the offset (first id of the chunk)
                offset = df[df["chunk_id"] == chunk].index[0]
                # apply a window and padding if necessary and required
                id = apply_window(
                    df[df["chunk_id"] == chunk],
                    self.seq_length,
                    self.min_seq_length,
                    self.continuous_window,
                    offset,
                    padding_id,
                )
                # combine all the sequences of the different chunks
                self.idx = torch.cat([self.idx, id])
            # summarize the chunks
            print("Data split per chunk:")
            print(f"Number of chunks: {len(self.chunk_lengths)}")
            print(
                f"Average chunk size: "
                f"{int(sum(self.chunk_lengths)/len(self.chunk_lengths))}"
            )
            print(f"Minimum chunk size: {min(self.chunk_lengths)}")
            print(f"Maximum chunk size: {max(self.chunk_lengths)}")

        else:
            # apply a window and padding if necessary and required
            self.idx = apply_window(
                df,
                self.seq_length,
                self.min_seq_length,
                self.continuous_window,
                0,
                padding_id,
            )

        # check if padding has been applied
        if self.min_seq_length < self.seq_length:
            # add trick for padding, add empty row add the end of the dataframe
            df = df.append(pd.Series(0, index=df.columns), ignore_index=True)
        self.dataset = df

    def split_dataset(self) -> None:
        id = np.arange(len(self.idx))
        # check in dataset must be split randomly
        if self.random:
            np.random.shuffle(id)
        # determine the length of each dataset
        train_len = np.floor(len(self.idx) * self.fractions[0]).astype(int)
        test_len = np.floor(len(self.idx) * self.fractions[1]).astype(int)
        valid_len = np.floor(len(self.idx) * self.fractions[2]).astype(int)
        # fill references of the train,test and validation data
        self.id_train = id[0:train_len]
        self.id_test = id[train_len: train_len + test_len]
        self.id_valid = id[train_len + test_len:
                           train_len + test_len + valid_len]

    def __len__(self) -> int:
        # returns the length of the idx, the number of sequences.
        return len(self.idx)

    def __getitem__(self, index: int) -> Tuple[Tensor]:
        # returns the lines of the dataframe
        # corresponding with the sequence (index)
        id = self.idx[index]
        data = self.dataset.loc[id, self.dataset.columns != "chunk_id"]
        return torch.Tensor(data.values)

    def batch(self, batchsize: int, target: str, single_cat: bool) -> Tensor:
        # based on the target select the train, test of validation set.
        if target == "train":
            id_subset = self.id_train
        elif target == "test":
            id_subset = self.id_test
        elif target == "valid":
            id_subset = self.id_valid
        else:
            print("Unknown input target")

        # determine how many batched can be created out
        # of the number of sequences in the dataset.
        nr_of_batches = int(np.floor(len(id_subset) / batchsize))
        batch_nr = 0
        while True:
            if batch_nr == nr_of_batches:
                batch_nr = 0
            seqX = []  # noqa N806
            seqY = []  # noqa N806
            for seq_nr in range(batchsize):
                # calculate the required id's
                id = id_subset[batch_nr * batchsize + seq_nr]
                # append the features of the sequences to the list
                seqX.append(
                    torch.Tensor(
                        self.dataset.loc[
                            self.idx[id],
                            ~self.dataset.columns.isin(["eyeDetection",
                                                        "chunk_id"]),
                        ].values
                    )
                )
                if single_cat:
                    # append the outcome of the sequences to the list
                    seqY.append(self.dataset.loc[int(self.idx[id][-1]),
                                                 "eyeDetection"])
                else:
                    # append the outcome of the sequences to the list
                    seqY.append(
                        torch.Tensor(
                            self.dataset.loc[self.idx[id],
                                             "eyeDetection"].values
                        )
                    )
            # stack the list of features
            X = torch.stack(seqX)  # noqa N806
            if type(seqY[0]) != Tensor:
                # resize the outcome to correspond with X
                Y = torch.Tensor(seqY)  # noqa N806
                Y = Y[:, None, None]  # noqa N806
            else:
                # resize the outcome to correspond with X
                Y = torch.stack(seqY)  # noqa N806
                Y = Y[:, :, None]  # noqa N806
            batch_nr = batch_nr + 1
            yield X, Y
