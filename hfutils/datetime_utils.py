import numpy as np

def np_datetime2file_datestr(time_np64: np.datetime64) -> str:
    format = '%Y%m%dT%H%M%SZ'
    return time_np64.astype('datetime64[us]').astype('O').strftime(format)