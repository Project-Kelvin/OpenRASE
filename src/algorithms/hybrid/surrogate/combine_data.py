"""
Defines functions to generate surrogate model training data.
"""

import os

import pandas as pd

from algorithms.hybrid.constants.surrogate import SURROGATE_DATA_PATH

OUTPUT: str = "latency"


def combineData() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generates data for training the surrogate model.
    """

    """
    Reads a CSV file and processes it into a DataFrame.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The training, test, and all data DataFrames.
    """

    filesToIgnore: "list[str]" = [
        "4_2.csv",
        "0.2_100.csv",
        "0.2_5.csv",
        "1_2.csv",
        "0.5_10.csv",
    ]
    files: "list[str]" = [
        f for f in os.listdir(SURROGATE_DATA_PATH) if f.endswith(".csv")
    ]
    files = [f for f in files if f not in filesToIgnore]
    data: list[pd.DataFrame] = [
        pd.read_csv(
            os.path.join(SURROGATE_DATA_PATH, f), sep=r"\s*,\s*", engine="python"
        )
        for f in files
    ]

    trainingData: pd.DataFrame = pd.DataFrame()
    testData: pd.DataFrame = pd.DataFrame()
    allData: pd.DataFrame = pd.DataFrame()
    for i, dataset in enumerate(data):
        dataset = dataset.dropna()
        dataset = dataset[dataset[OUTPUT] != 0]
        dataset[OUTPUT] = dataset[OUTPUT] - dataset["total_delay"]  # Deduct delay
        allDataset: pd.DataFrame = pd.DataFrame()
        trainingDataset: pd.DataFrame = pd.DataFrame()
        testDataset: pd.DataFrame = pd.DataFrame()

        for gen, genData in dataset.groupby("generation"):
            binSize: int = 10
            genData = genData.groupby(genData.index // binSize).agg(
                latency=("latency", "median"),
                max_cpu=("max_cpu", "median"),
                total_link_score=("total_link_score", "median"),
                max_link_score=("max_link_score", "median"),
                total_delay=("total_delay", "median"),
            )

            q1: float = genData[OUTPUT].quantile(0.25)
            q3: float = genData[OUTPUT].quantile(0.75)
            iqr: float = q3 - q1
            lowerBound: float = q1 - 1.5 * iqr
            upperBound: float = q3 + 1.5 * iqr

            genData = genData[
                (genData[OUTPUT] > lowerBound) & (genData[OUTPUT] < upperBound)
            ]

            dataToTrain: pd.DataFrame = genData.sample(frac=0.8, random_state=0)
            allData = pd.concat([allData, genData])
            trainingData = pd.concat([trainingData, dataToTrain])
            testData = pd.concat([testData, genData.drop(dataToTrain.index)])

    return trainingData, testData, allData
