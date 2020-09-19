import pandas as pd


NAMES_PATHS = [
    "dns/dns2tcp/names.csv",
    "dns/dnscapy/names.csv",
    "dns/iodine/names.csv",
    "dns/plain/names.csv",
    "dns/tuns/names.csv",
    "../DomainFrontingLists/Google-hosted.txt",
    "../DomainFrontingLists/Cloudfront.txt",
]
TRAIN_PATHS = [
    "train.csv",
    "validate.csv",
]


names_df = pd.DataFrame([], columns=["label", "qname"])

for path in NAMES_PATHS:
    df = pd.read_csv(path)
    names_df = names_df.append(df)


train_df = pd.DataFrame([], columns=["label", "qname"])
train_df = train_df.set_index("qname")

for path in TRAIN_PATHS:
    df = pd.read_csv(f"dns/{path}", header=None, names=["label", "qname"])
    train_df = train_df.append(df)


test_df = names_df[~names_df["qname"].isin(train_df["qname"])]
test_df = test_df.sample(20000)
test_df.to_csv("dns/test.csv", index=False, header=False)
