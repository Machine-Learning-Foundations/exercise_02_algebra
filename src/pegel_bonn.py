"""Analysis of the Rhine-water level data from https://pegel.bonn.de/php/rheinpegel.php ."""

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas


if __name__ == "__main__":
    rhein = pandas.read_csv("./data/pegel.tab", sep=" 	")
    levels = np.array([int(pegel.split(" ")[0]) for pegel in rhein["Pegel"]])

    timestamps = [ts[:-4] for ts in rhein["Zeit"]]
    datetime_list = []
    keep_indices = []
    for idx, ts in enumerate(timestamps):
        ts_date, ts_time = ts.split(",")
        day, month, year = ts_date.split(".")
        hour, minute = ts_time.split(":")
        datetime_list.append(datetime(int(year), int(month), int(day)))

    # uncomment to start in the year 2000.
    # levels = levels[:-30_000:4]
    # datetime_list = datetime_list[:-30_000:4]

    datetime_stamps = [dt.timestamp() for dt in datetime_list]

    levels = levels[::-1]
    datetime_stamps = datetime_stamps[::-1]
    datetime_list = datetime_list[::-1]

    plt.plot(datetime_list, levels)
    plt.xlabel("Year")
    plt.ylabel("Water level [cm]")
    plt.title("Rhine water level in Bonn")
    plt.show()
