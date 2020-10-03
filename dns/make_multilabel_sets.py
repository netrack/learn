from functools import reduce
import pandas as pd


PATHS = [
    "dns/dns2tcp/2018-03-23-11-08-11.csv",
    "dns/dnscapy/2018-03-29-19-06-25.csv",
    "dns/iodine/2018-03-19-19-06-24.csv",
    "dns/plain/2018-03-19-19-34-33.csv",
    "dns/tuns/2018-03-30-09-40-10.csv",
]


dataframes = [pd.read_csv(path) for path in PATHS]
mincount = min([df.shape[0] for df in dataframes])

# Crop the dataframes to the same same, to make a uniform sampling.
dataframes = [df.sample(mincount) for df in dataframes]
names_df = reduce(lambda x, y: x.append(y), dataframes)

train_df = names_df.sample(16000, random_state=1)
validate_df = names_df.sample(5000, random_state=2)
test_df = names_df.sample(20000, random_state=3)

train_df.to_csv("multilabel/train.csv", index=False, header=True)
validate_df.to_csv("multilabel/validate.csv", index=False, header=True)
test_df.to_csv("multilabel/test.csv", index=False, header=True)
