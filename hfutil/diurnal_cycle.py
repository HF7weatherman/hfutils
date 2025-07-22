import numpy as np
import xarray as xr

# ------------------------------------------------------------------------------
# Functions for calculating area-averaged diurnal cycles
# ------------------------------------------------------------------------------
def get_approx_localtime(
    reference_datetime: xr.DataArray,
    longitude: xr.DataArray,
    time_resolution_seconds: int,
    keep_time_resolution: bool=True,
    center: bool=False,
    ) -> xr.DataArray:
    """
    Computes an approximation of the local time based on the provided datetime
    and longitude.

    Parameters
    ----------
    reference_datetime : xr.DataArray
        Array of datetime values for which the local time approximation is
        computed.
    longitude : xr.DataArray
        Array of longitude values corresponding to the datetime values.
    time_resolution_seconds : int
        The time resolution in seconds of the input dataset.
    keep_time_resolution : bool, optional
        If True, maintains the granularity of the time resolution.
        Default is True.
    center : bool, optional
        If True, centers the local time offset within the time resolution
        interval. Default is False.

    Returns
    -------
    xr.DataArray
        Array of datetime values adjusted to approximate the local time.
    """
    local_time_offset = ((longitude * 12/180) * 60**2).astype('timedelta64[s]')

    if keep_time_resolution:
        local_time_offset = local_time_offset.astype(
            f'timedelta64[{time_resolution_seconds}s]'
            ) * time_resolution_seconds

        if center:
            local_time_offset = local_time_offset + \
                np.array(time_resolution_seconds/2).astype('timedelta64[s]')

    return reference_datetime + local_time_offset


def _get_time_resolution(time: xr.DataArray) -> int:
    """
    Determine the time resolution of a dataset.

    This function calculates the unique time difference between consecutive 
    samples in the dataset along the 'time' dimension. If the time differences 
    are not uniform, an error is raised.

    Parameters
    ----------
    time : xarray.DataArray
        A DataArray containing the time dimension of the dataset.

    Returns
    -------
    int
        The time resolution of the dataset in seconds.

    Raises
    ------
    ValueError
        If the time samples are not evenly spaced.
    """
    unique_time_dels = np.unique(time.diff(dim='time')).astype('timedelta64[s]')
    if len(unique_time_dels) > 1:
        raise ValueError(
            "The samples of the dataset are not evenly spaced in time. Please" +
            " doublecheck and provide a dataset with evenly spaced samples!"
            )
    else:
        return unique_time_dels[0].astype('int')


def avg_diurnal_cycle(dset: xr.Dataset, **kwargs) -> xr.Dataset:
    """
    Compute the average diurnal cycle of a dataset by grouping data based on 
    approximate local time.
    This function stacks the longitude and time dimensions into a single 
    dimension called 'local_time', assigns approximate local time coordinates 
    to the dataset, and then computes the mean for each local time group.

    Parameters
    ----------
    dset : xarray.Dataset
        Input dataset containing 'time' and 'lon' dimensions.
    **kwargs : dict, optional
        Additional keyword arguments passed to the `_localTimeApprox` function 
        for customizing the local time approximation.

    Returns
    -------
    xarray.Dataset
        Dataset with the average diurnal cycle grouped by approximate local time.
    """
    time_resolution_seconds = _get_time_resolution(dset['time'])

    dset = dset.stack(local_time=('lon', 'time'))
    dset = dset.assign_coords(
        local_time = get_approx_localtime(
            dset['time'], dset['lon'],
            time_resolution_seconds=time_resolution_seconds, **kwargs,
            )
            ).squeeze()

    return dset.groupby("local_time.time").mean(skipna=True)