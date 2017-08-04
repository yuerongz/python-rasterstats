# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
import numpy as np
import warnings
from affine import Affine
from shapely.geometry import shape
from .io import read_features, Raster
from .utils import (rasterize_geom, get_percentile, check_stats,
                    remap_categories, key_assoc_val, boxify_points,
                    rasterize_pctcover_geom, get_latitude_scale,
                    split_geom, VALID_STATS)


def raster_stats(*args, **kwargs):
    """Deprecated. Use zonal_stats instead."""
    warnings.warn("'raster_stats' is an alias to 'zonal_stats'"
                  " and will disappear in 1.0", DeprecationWarning)
    return zonal_stats(*args, **kwargs)


def zonal_stats(*args, **kwargs):
    """The primary zonal statistics entry point.

    All arguments are passed directly to ``gen_zonal_stats``.
    See its docstring for details.

    The only difference is that ``zonal_stats`` will
    return a list rather than a generator."""
    return list(gen_zonal_stats(*args, **kwargs))


def gen_zonal_stats(
        vectors, raster,
        layer=0,
        band=1,
        nodata=None,
        affine=None,
        stats=None,
        all_touched=False,
        latitude_correction=False,
        percent_cover_selection=None,
        percent_cover_weighting=False,
        percent_cover_scale=None,
        limit=None,
        categorical=False,
        category_map=None,
        add_stats=None,
        zone_func=None,
        raster_out=False,
        prefix=None,
        geojson_out=False, **kwargs):
    """Zonal statistics of raster values aggregated to vector geometries.

    Parameters
    ----------
    vectors: path to an vector source or geo-like python objects

    raster: ndarray or path to a GDAL raster source
        If ndarray is passed, the ``affine`` kwarg is required.

    layer: int or string, optional
        If `vectors` is a path to an fiona source,
        specify the vector layer to use either by name or number.
        defaults to 0

    band: int, optional
        If `raster` is a GDAL source, the band number to use (counting from 1).
        defaults to 1.

    nodata: float, optional
        If `raster` is a GDAL source, this value overrides any NODATA value
        specified in the file's metadata.
        If `None`, the file's metadata's NODATA value (if any) will be used.
        defaults to `None`.

    affine: Affine instance
        required only for ndarrays, otherwise it is read from src

    stats:  list of str, or space-delimited str, optional
        Which statistics to calculate for each zone.
        All possible choices are listed in ``utils.VALID_STATS``.
        defaults to ``DEFAULT_STATS``, a subset of these.

    all_touched: bool, optional
        Whether to include every raster cell touched by a geometry, or only
        those having a center point within the polygon.
        defaults to `False`

    latitude_correction: bool, optional
        * For use with WGS84 data only.
        * Only applies to "mean" stat.
        Weights cell values when generating statistics based on latitude
        (using haversine function) in order to account for actual area
        represented by pixel cell.
    percent_cover_selection: float, optional
        Include only raster cells that have at least the given percent
        covered by the vector feature. Requires percent_cover_scale argument
        be used to specify scale at which to generate percent coverage
        estimates

    percent_cover_weighting: bool, optional
        whether or not to use percent coverage of cells during calculations
        to adjust stats (only applies to mean, count and sum)

    percent_cover_scale: int, optional
        Scale used when generating percent coverage estimates of each
        raster cell by vector feature. Percent coverage is generated by
        rasterizing the feature at a finer resolution than the raster
        (based on percent_cover_scale value) then using a summation to aggregate
        to the raster resolution and dividing by the square of percent_cover_scale
        to get percent coverage value for each cell. Increasing percent_cover_scale
        will increase the accuracy of percent coverage values; three orders
        magnitude finer resolution (percent_cover_scale=1000) is usually enough to
        get coverage estimates with <1% error in individual edge cells coverage
        estimates, though much smaller values (e.g., percent_cover_scale=10) are often
        sufficient (<10% error) and require less memory.

    limit: int
        maximum number of pixels allowed to be read from raster based on
        feature bounds. Geometries which will result in reading a larger
        number of pixels will be split into smaller geometries and then
        aggregated (note: some stats and options cannot be used along with
        `limit`. Useful when dealing with vector data containing
        large features and raster with a fine resolution to prevent
        memory errors. Estimated pixels per GB vary depending on options,
        but a rough range is 5 to 80 million pixels per GB of memory. If
        values is None (default) geometries will never be split. Using the
        `limit` option without also using the `percent_cover_weighting`
        option may result in less accurate statistics due to generating
        additional polygon edges when splitting geometries.

    categorical: bool, optional

    category_map: dict
        A dictionary mapping raster values to human-readable categorical names.
        Only applies when categorical is True

    add_stats: dict
        with names and functions of additional stats to compute, optional

    zone_func: callable
        function to apply to zone ndarray prior to computing stats

    raster_out: boolean
        Include the masked numpy array for each feature?, optional

        Each feature dictionary will have the following additional keys:
        mini_raster_array: The clipped and masked numpy array
        mini_raster_affine: Affine transformation
        mini_raster_nodata: NoData Value

    prefix: string
        add a prefix to the keys (default: None)

    geojson_out: boolean
        Return list of GeoJSON-like features (default: False)
        Original feature geometry and properties will be retained
        with zonal stats appended as additional properties.
        Use with `prefix` to ensure unique and meaningful property names.

    Returns
    -------
    generator of dicts (if geojson_out is False)
        Each item corresponds to a single vector feature and
        contains keys for each of the specified stats.

    generator of geojson features (if geojson_out is True)
        GeoJSON-like Feature as python dict
    """
    stats, run_count = check_stats(stats, categorical)

    # Handle 1.0 deprecations
    transform = kwargs.get('transform')
    if transform:
        warnings.warn("GDAL-style transforms will disappear in 1.0. "
                      "Use affine=Affine.from_gdal(*transform) instead",
                      DeprecationWarning)
        if not affine:
            affine = Affine.from_gdal(*transform)

    cp = kwargs.get('copy_properties')
    if cp:
        warnings.warn("Use `geojson_out` to preserve feature properties",
                      DeprecationWarning)

    band_num = kwargs.get('band_num')
    if band_num:
        warnings.warn("Use `band` to specify band number", DeprecationWarning)
        band = band_num


    # -------------------------------------------------------------------------
    # make sure feature split/aggregations will work with options provided

    invalid_limit_stats = [
        'minority', 'majority', 'median', 'std', 'unique'
    ] + [s for s in stats if s.startswith('percentile_')]

    invalid_limit_conditions = (
        any([i in invalid_limit_stats for i in stats])
        or add_stats is not None
        or raster_out
    )
    if limit is not None and invalid_limit_conditions:
        raise Exception("Cannot use `limit` to split geometries when using "
                        "`add_stats` or `raster_out` options")

    if limit is not None and all_touched and not percent_cover_weighting:
        warnings.warn('Using the `limit` option to split large geometries '
                      'along with the `all_touched` options without also '
                      'using the `percent_cover_weighting` option may result '
                      'in less accurate statistics')


    # -------------------------------------------------------------------------
    # check inputs related to percent coverage
    percent_cover = False
    if percent_cover_weighting or percent_cover_selection is not None:
        percent_cover = True
        if percent_cover_scale is None:
            warnings.warn('No value for `percent_cover_scale` was given. '
                          'Using default value of 10.')
            percent_cover_scale = 10

        try:
            if percent_cover_scale != int(percent_cover_scale):
                warnings.warn('Value for `percent_cover_scale` given ({0}) '
                              'was converted to int ({1}) but does not '
                              'match original value'.format(
                                percent_cover_scale, int(percent_cover_scale)))

            percent_cover_scale = int(percent_cover_scale)

            if percent_cover_scale <= 1:
                raise Exception('Value for `percent_cover_scale` must be '
                                'greater than one ({0})'.format(
                                    percent_cover_scale))

        except:
            raise Exception('Invalid value for `percent_cover_scale` '
                            'provided ({0}). Must be type int.'.format(
                                percent_cover_scale))

        if percent_cover_selection is not None:
            try:
                percent_cover_selection = float(percent_cover_selection)
            except:
                raise Exception('Invalid value for `percent_cover_selection` '
                                'provided ({0}). Must be able to be converted '
                                'to a float.'.format(percent_cover_selection))

        # if not all_touched:
        #     warnings.warn('`all_touched` was not enabled but an option requiring '
        #                   'percent_cover calculations was selected. Automatically '
        #                   'enabling `all_touched`.')
        # all_touched = True


    with Raster(raster, affine, nodata, band) as rast:
        features_iter = read_features(vectors, layer)
        for _, feat in enumerate(features_iter):
            geom = shape(feat['geometry'])

            if 'Point' in geom.type:
                geom = boxify_points(geom, rast)
                percent_cover = False

            geom_bounds = tuple(geom.bounds)


            # -----------------------------------------------------------------
            # build geom_list (split geoms if needed)


            if limit is None:
                geom_list = [geom]

            else:
                # need count for sub geoms to calculate weighted mean
                if 'mean' in stats and not 'count' in stats:
                    stats.append('count')
                pixel_size = rast.affine[0]
                x_size = (geom_bounds[2] - geom_bounds[0]) / pixel_size
                y_size = (geom_bounds[3] - geom_bounds[1]) / pixel_size
                total_size = x_size * y_size

                geom_list = split_geom(geom, limit, pixel_size)


            # -----------------------------------------------------------------
            # run sub geom extracts

            sub_feature_stats_list = []

            for sub_geom in geom_list:

                sub_geom = shape(sub_geom)

                if 'Point' in sub_geom.type:
                    sub_geom = boxify_points(sub_geom, rast)

                sub_geom_bounds = tuple(sub_geom.bounds)

                fsrc = rast.read(bounds=sub_geom_bounds)

                # rasterized geometry
                if percent_cover:
                    cover_weights = rasterize_pctcover_geom(
                        sub_geom, shape=fsrc.shape, affine=fsrc.affine,
                        scale=percent_cover_scale,
                        all_touched=all_touched)
                    rv_array = cover_weights > (percent_cover_selection or 0)
                else:
                    rv_array = rasterize_geom(
                        sub_geom, shape=fsrc.shape, affine=fsrc.affine,
                        all_touched=all_touched)

                # nodata mask
                isnodata = (fsrc.array == fsrc.nodata)

                # add nan mask (if necessary)
                has_nan = (np.issubdtype(fsrc.array.dtype, float)
                    and np.isnan(fsrc.array.min()))
                if has_nan:
                    isnodata = (isnodata | np.isnan(fsrc.array))

                # Mask the source data array
                # mask everything that is not a valid value or not within our geom

                masked = np.ma.MaskedArray(
                    fsrc.array,
                    mask=(isnodata | ~rv_array))


                # execute zone_func on masked zone ndarray
                if zone_func is not None:
                    if not callable(zone_func):
                        raise TypeError(('zone_func must be a callable '
                                         'which accepts function a '
                                         'single `zone_array` arg.'))
                    zone_func(masked)

                if latitude_correction and 'mean' in stats:
                    latitude_scale = [
                        get_latitude_scale(fsrc.affine[5] - abs(fsrc.affine[4]) * (0.5 + i))
                        for i in range(fsrc.shape[0])
                    ]

                if masked.compressed().size == 0:
                    # nothing here, fill with None and move on
                    sub_feature_stats = dict([(stat, None) for stat in stats])
                    if 'count' in stats:  # special case, zero makes sense here
                        sub_feature_stats['count'] = 0
                else:
                    if run_count:
                        keys, counts = np.unique(masked.compressed(), return_counts=True)
                        pixel_count = dict(zip([np.asscalar(k) for k in keys],
                                               [np.asscalar(c) for c in counts]))

                    if categorical:
                        sub_feature_stats = dict(pixel_count)
                        if category_map:
                            sub_feature_stats = remap_categories(category_map, sub_feature_stats)
                    else:
                        sub_feature_stats = {}

                    if 'min' in stats:
                        sub_feature_stats['min'] = float(masked.min())
                    if 'max' in stats:
                        sub_feature_stats['max'] = float(masked.max())

                    if 'mean' in stats:
                        if percent_cover_weighting and latitude_correction:
                            sub_feature_stats['mean'] = float(
                                np.sum((masked.T * latitude_scale).T * cover_weights) /
                                np.sum((~masked.mask.T * latitude_scale).T * cover_weights))
                        elif percent_cover_weighting:
                            sub_feature_stats['mean'] = float(
                                np.sum(masked * cover_weights) /
                                np.sum(~masked.mask * cover_weights))
                        elif latitude_correction:
                            sub_feature_stats['mean'] = float(
                                np.sum((masked.T * latitude_scale).T) /
                                np.sum(latitude_scale *
                                       (masked.shape[1] - np.sum(masked.mask, axis=1))))
                        else:
                            sub_feature_stats['mean'] = float(masked.mean())

                    if 'count' in stats:
                        if percent_cover_weighting:
                            sub_feature_stats['count'] = float(np.sum(~masked.mask * cover_weights))
                        else:
                            sub_feature_stats['count'] = int(masked.count())

                    if 'sum' in stats:
                        if percent_cover_weighting:
                            sub_feature_stats['sum'] = float(np.sum(masked * cover_weights))
                        else:
                            sub_feature_stats['sum'] = float(masked.sum())

                    if 'std' in stats:
                        sub_feature_stats['std'] = float(masked.std())
                    if 'median' in stats:
                        sub_feature_stats['median'] = float(np.median(masked.compressed()))
                    if 'majority' in stats:
                        sub_feature_stats['majority'] = float(key_assoc_val(pixel_count, max))
                    if 'minority' in stats:
                        sub_feature_stats['minority'] = float(key_assoc_val(pixel_count, min))
                    if 'unique' in stats:
                        sub_feature_stats['unique'] = len(list(pixel_count.keys()))
                    if 'range' in stats:
                        rmin = float(masked.min())
                        rmax = float(masked.max())
                        sub_feature_stats['min'] = rmin
                        sub_feature_stats['max'] = rmax
                        sub_feature_stats['range'] = rmax - rmin

                    for pctile in [s for s in stats if s.startswith('percentile_')]:
                        q = get_percentile(pctile)
                        pctarr = masked.compressed()
                        sub_feature_stats[pctile] = np.percentile(pctarr, q)

                if 'nodata' in stats or 'nan' in stats:
                    featmasked = np.ma.MaskedArray(fsrc.array, mask=(~rv_array))

                    if 'nodata' in stats:
                        sub_feature_stats['nodata'] = float((featmasked == fsrc.nodata).sum())
                    if 'nan' in stats:
                        sub_feature_stats['nan'] = float(np.isnan(featmasked).sum()) if has_nan else 0

                if add_stats is not None:
                    for stat_name, stat_func in add_stats.items():
                        sub_feature_stats[stat_name] = stat_func(masked)

                if raster_out:
                    sub_feature_stats['mini_raster_array'] = masked
                    sub_feature_stats['mini_raster_affine'] = fsrc.affine
                    sub_feature_stats['mini_raster_nodata'] = fsrc.nodata

                sub_feature_stats_list.append(sub_feature_stats)


            # -----------------------------------------------------------------
            # aggregate sub geom extracts

            if len(geom_list) == 1:

                feature_stats = sub_feature_stats_list[0]

                if 'range' in stats and not 'min' in stats:
                    del feature_stats['min']
                if 'range' in stats and not 'max' in stats:
                    del feature_stats['max']

            else:
                feature_stats = {}

                if 'count' in stats:
                    feature_stats['count'] = sum([i['count'] for i in sub_feature_stats_list])
                if 'min' in stats:
                    vals = [i['min'] for i in sub_feature_stats_list if i['min'] is not None]
                    feature_stats['min'] = min(vals) if vals else None
                if 'max' in stats:
                    feature_stats['max'] = max([i['max'] for i in sub_feature_stats_list])
                if 'range' in stats:
                    vals = [i['min'] for i in sub_feature_stats_list if i['min'] is not None]
                    rmin = min(vals) if vals else None
                    rmax = max([i['max'] for i in sub_feature_stats_list])
                    feature_stats['range'] = rmax - rmin if rmin is not None else None
                if 'mean' in stats:
                    vals = [i['mean'] * i['count'] for i in sub_feature_stats_list if i['mean'] is not None]
                    feature_stats['mean'] = sum(vals) / sum([i['count'] for i in sub_feature_stats_list]) if vals else None
                if 'sum' in stats:
                    vals = [i['sum'] for i in sub_feature_stats_list if i['sum'] is not None]
                    feature_stats['sum'] = sum(vals) if vals else None
                if 'nodata' in stats:
                    feature_stats['nodata'] = sum([i['nodata'] for i in sub_feature_stats_list])
                if 'nan' in stats:
                    feature_stats['nan'] = sum([i['nan'] for i in sub_feature_stats_list])
                if categorical:
                    for sub_stats in sub_feature_stats_list:
                        for field in sub_stats:
                            if field not in VALID_STATS:
                                if field not in feature_stats:
                                    feature_stats[field] = sub_stats[field]
                                else:
                                    feature_stats[field] += sub_stats[field]



            if prefix is not None:
                prefixed_feature_stats = {}
                for key, val in feature_stats.items():
                    newkey = "{}{}".format(prefix, key)
                    prefixed_feature_stats[newkey] = val
                feature_stats = prefixed_feature_stats

            if geojson_out:
                for key, val in feature_stats.items():
                    if 'properties' not in feat:
                        feat['properties'] = {}
                    feat['properties'][key] = val
                yield feat
            else:
                yield feature_stats
