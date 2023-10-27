import tiktoken
import datasets

if __name__ == "__main__":
    enc = tiktoken.get_encoding("cl100k_base")
    enc = tiktoken.encoding_for_model("gpt-4")

    imdb = datasets.load_dataset("imdb")

    train = imdb["train"].to_pandas()
    test = imdb["test"].to_pandas()

    train.to_csv("data/raw/imdb/train.csv", index=False)
    test.to_csv("data/raw/imdb/test.csv", index=False)

    train["token_size"] = train["text"].apply(lambda x: len(enc.encode(x)))

    median = train["token_size"].median()
    print(f"Median token size: {median}")

    train_sample = train[(train["token_size"] > median - 10) & (train["token_size"] < median + 10)]
    train_sample = train_sample.groupby("label").apply(lambda x: x.sample(5, random_state=42)).reset_index(drop=True)

    # Sample a dev set of 1000 examples 
    dev_sample = train.groupby("label").apply(lambda x: x.sample(500, random_state=42)).reset_index(drop=True)

    train_sample.drop(columns=["token_size"], inplace=True)
    train_sample.rename(columns={"text": "review"}, inplace=True)
    train_sample.rename(columns={"Label": "label"}, inplace=True)

    train_sample.to_csv("data/processed/imdb/train.csv", index=False)

    dev_sample.drop(columns=["token_size"], inplace=True)
    dev_sample.rename(columns={"Label": "label"}, inplace=True)

    dev_sample.to_csv("data/processed/imdb/dev.csv", index=False)