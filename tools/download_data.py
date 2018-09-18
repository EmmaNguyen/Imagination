import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Downloader of 0.1 \
        preprocessed data - ShepardMetzler 7 parts.")

    parser.add_argument("--data_path", type=str, default="./data", \
        help="A path to directory of data to be downloaded.")

    return parser.parse_args()

def download_data(data_path):
    file_name = "torch_small_shepard_metzler.tar"
    download_url="https://www.dropbox.com/s/fc16aeeo0014dv2/{}".format(file_name)
    test_link = "https://www.dropbox.com/s/fc16aeeo0014dv2/torch_small_shepard_metzler.tar?dl=0"
    try:
        import wget
        wget.download(test_link, out=".")
    except:
        print("Require to install package: try `conda install wget`. More information at https://pypi.org/project/wget/")

if __name__=="__main__":
    args = parse_args()
    download_data(args.data_path)
