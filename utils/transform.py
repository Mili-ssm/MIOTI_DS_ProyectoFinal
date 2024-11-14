import json
import math
import os
from dataclasses import asdict, dataclass
from enum import auto
from functools import lru_cache
from typing import Self

import numpy as np
import polars as pl
from geopy.geocoders import Nominatim
from zmq import IntEnum

type DataFrame = pl.DataFrame | pl.LazyFrame


def cyclic_transform[T: DataFrame](
    data: T,
    column: str,
    max_val: float,
    fill_nan: bool = True,
) -> T:
    """Transform a column to a cyclic representation."""
    pi_2 = 2 * np.pi
    col = pl.col(column)
    new_data = data.with_columns(
        ((col * pi_2) / max_val).sin().alias(column + "_sin"),
        ((col * pi_2) / max_val).cos().alias(column + "_cos"),
    )

    if fill_nan:
        col_sin = pl.col(column + "_sin")
        col_cos = pl.col(column + "_cos")
        new_data = new_data.with_columns(
            col_sin.fill_null(col_sin.drop_nans().drop_nulls().mean()),
            col_cos.fill_null(col_cos.drop_nans().drop_nulls().mean()),
        )

    return new_data.drop(column)  # type: ignore


def cyclic_transform_rads[T: DataFrame](
    data: T,
    column: str,
    fill_nan: bool = True,
) -> T:
    """Transform a column to a cyclic representation."""
    col = pl.col(column)
    new_data = data.with_columns(
        ((col * np.pi) / 180).sin().alias(column + "_sin"),
        ((col * np.pi) / 180).cos().alias(column + "_cos"),
    )

    if fill_nan:
        col_sin = pl.col(column + "_sin")
        col_cos = pl.col(column + "_cos")
        new_data = new_data.with_columns(
            col_sin.fill_null(col_sin.drop_nans().drop_nulls().mean()),
            col_cos.fill_null(col_cos.drop_nans().drop_nulls().mean()),
        )

    return new_data.drop(column)  # type: ignore


class DateGranularity(IntEnum):
    """Date granularity.

    The granularity of the date transformations.
    """

    MONTHS = auto()
    """ Only the month of the year. """
    DAYS = auto()
    """ The day of the year, the day of the month and the day of the week. """
    HOURS = auto()
    """ The hour of the day. """

    @staticmethod
    def all() -> list["DateGranularity"]:
        """Return all the date granularities."""
        return list(DateGranularity)


def date_to_cyclic(
    data: pl.DataFrame,
    column: str,
    fill_nan: bool = True,
    granularity: list[DateGranularity] = DateGranularity.all(),
) -> pl.DataFrame:
    """Transform a date column to a cyclic representation."""
    col_dt = pl.col(column).dt
    new_data = data.lazy()

    if DateGranularity.HOURS in granularity:
        new_data = new_data.with_columns(
            (col_dt.hour() + (col_dt.minute() / 60)).alias(column + "_hour_of_day")
        )
        new_data = cyclic_transform(new_data, column + "_hour_of_day", 24, fill_nan=fill_nan)

    if DateGranularity.DAYS in granularity:
        new_data = new_data.with_columns(col_dt.weekday().alias(column + "_day_of_week"))
        new_data = cyclic_transform(new_data, column + "_day_of_week", 7, fill_nan=fill_nan)

        new_data = new_data.with_columns(col_dt.day().alias(column + "_day_of_month"))
        new_data = cyclic_transform(new_data, column + "_day_of_month", 31, fill_nan=fill_nan)

    if DateGranularity.MONTHS in granularity and DateGranularity.DAYS in granularity:
        new_data = new_data.with_columns(col_dt.ordinal_day().alias(column + "_day_of_year"))
        new_data = cyclic_transform(new_data, column + "_day_of_year", 365, fill_nan=fill_nan)

    if DateGranularity.MONTHS in granularity:
        new_data = new_data.with_columns(col_dt.month().alias(column + "_month_of_year"))
        new_data = cyclic_transform(new_data, column + "_month_of_year", 12, fill_nan=fill_nan)

    return new_data.drop(column).collect()


conector = Nominatim(user_agent="buscador_de_ubicaciones_v1")

__cache__ = {}


@dataclass
class GeoPoint:
    latitude: float | None
    longitude: float | None

    @classmethod  # type: ignore
    @lru_cache(maxsize=128)
    def get_geopoint(cls, *addres: str, cache_file: str = "./geopoints.json") -> Self:
        global __cache__
        full_addres = " ".join(addres)

        if not __cache__:
            if os.path.exists(cache_file):
                with open(cache_file, encoding="utf-8") as f:
                    __cache__ = json.load(f)
                    cache = __cache__
            else:
                cache = {}
        else:
            cache = __cache__

        value = cache.get(full_addres)

        if value is None:
            return cls(latitude=None, longitude=None)

        return cls(
            latitude=float(value["latitude"]),
            longitude=float(value["longitude"]),
        )

        location = conector.geocode(full_addres)  # type: ignore
        if location is not None:
            lat_rad = math.radians(location.latitude)
            lon_rad = math.radians(location.longitude)
        else:
            lat_rad = 0
            lon_rad = 0

        geo_point = cls(
            latitude=lat_rad,
            longitude=lon_rad,
        )

        cache[full_addres] = asdict(geo_point)

        with open(cache_file, mode="w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=4)

        return geo_point


def location_to_cyclic(
    data: pl.DataFrame,
    column: str,
    *pre_fix: str,
    fill_nan: bool = True,
) -> pl.DataFrame:
    """Transform a location column to a cyclic representation."""
    new_data = data.lazy().with_columns(
        pl.col(column)
        .map_elements(
            lambda x: GeoPoint.get_geopoint(*pre_fix, x).latitude, return_dtype=pl.Float64
        )
        .alias(column + "_lat"),
        pl.col(column)
        .map_elements(
            lambda x: GeoPoint.get_geopoint(*pre_fix, x).longitude, return_dtype=pl.Float64
        )
        .alias(column + "_lon"),
    )

    new_data = cyclic_transform_rads(new_data, column + "_lat", fill_nan=fill_nan)
    new_data = cyclic_transform_rads(new_data, column + "_lon", fill_nan=fill_nan)

    return new_data.drop(column).collect()


def int_to_cp(txt: str) -> str:
    while len(txt) < 5:
        txt = "0" + txt
    return txt
