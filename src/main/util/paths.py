import os

# src/main/util
UTIL = os.path.dirname(os.path.abspath(__file__))

# src/main
MAIN = os.path.dirname(UTIL)

# src
SRC = os.path.dirname(MAIN)

# src/data
DATA = os.path.join(SRC, "data")

# src/data/raw
RAW_DATA = os.path.join(DATA, "raw")

PROC_DATA = os.path.join(DATA, "processed")

# Ensure the folders exist
os.makedirs(RAW_DATA, exist_ok=True)
os.makedirs(PROC_DATA, exist_ok=True)


if __name__ == "__main__":
    print("UTIL_DIR:", UTIL)
    print("MAIN_DIR:", MAIN)
    print("SRC_DIR:", SRC)
    print("DATA_DIR:", DATA)
    print("RAW_DATA_DIR:", RAW_DATA)