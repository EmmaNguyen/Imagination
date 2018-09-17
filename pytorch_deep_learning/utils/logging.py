import datetime

def get_finish_time(format="%d_%m_%Y_%H_%M_%S"):
    return datetime.datetime.now().strftime(format)
