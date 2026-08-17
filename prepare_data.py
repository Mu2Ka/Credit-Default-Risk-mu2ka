from data_loader import *
from features import *
from preprocessing import *
## data_loader  - получили test,train агрегированный по id
##features - агрегировали
## preprocessing onehot и тд

def prepare_data(data_dir: str = "data/raw"):
    train, test = build_modeling_tables(data_dir)
    train = FEATURE_ENGINEERING(train)
    test = FEATURE_ENGINEERING(test)
    train.to_csv("data/processed/train_processed.csv", index=False)
    test.to_csv("data/processed/test_processed.csv", index=False)
if __name__ == "__main__":
    prepare_data()
