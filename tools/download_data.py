import argsparse

def parse_args():
    parser = argsparse.ArgumentParser(description="Downloader of 0.1 \
        preprocessed data - ShepardMetzler 7 parts.")

    parser.add_argument("--data_path", type=str, default="./data", \
        help="A path to directory of data to be downloaded.")

    return parser.parse_args()

def download_data():
    pass

if __name__=="__main__":
    args = parse_args()
    download_data(args.data_path)
