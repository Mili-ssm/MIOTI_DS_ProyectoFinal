"""Common data tools for data manipulation."""

from collections.abc import Iterable, Sequence
from dataclasses import InitVar, dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Self
from uuid import uuid4

import altair as alt
import polars as pl
import polars.selectors as cs
import seaborn as sns
from IPython.display import display
from matplotlib import pyplot as plt
from polars._typing import IntoExpr


def get_feature_importance(
    features: Iterable[str],
    importances: Iterable[float],
    percentiles: list[int] = [50, 75, 90, 95, 99, 100],
) -> pl.DataFrame:
    def select_percentil(x: float):
        for percentil in percentiles:
            if x <= percentil:
                return f"{percentil:03d}"
        return f"{percentiles[-1]:03d}"

    feature_importances = (
        pl.DataFrame({
            "Feature": features,
            "Importance": importances,
        })
        .with_columns(
            pl.col("Importance") * 100,
        )
        .sort(
            by="Importance",
            descending=True,
        )
    )

    # Mostrar el DataFrame
    display(feature_importances)

    feature_importances.with_columns(
        pl.col("Importance")
        .cum_sum()
        .map_elements(
            select_percentil,
            pl.String,
        )
        .alias("percentil")
    ).plot.bar(
        y=alt.Y("Importance").scale(domain=(0, 100)),
        x=alt.X("Feature", sort="-y"),
        color="percentil",
    ).show()

    return feature_importances


class Show(Enum):
    """Enum for the show parameter in the data tools."""

    PLOT = auto()
    DATA = auto()
    TITLE = auto()
    NONE = auto()


@dataclass(slots=True, frozen=True)
class Splitter:
    """Splitter class for splitting Complex String Columns."""

    col: str
    sep: str
    names: Sequence[str]


@dataclass(slots=True)
class CustomDataFrame:
    """Custom data frame class for High-Level data manipulation."""

    data_paths: InitVar[Iterable[Path | str]]
    separator: InitVar[str] = field(default=",")

    hidden_cols: list[str] = field(default_factory=list, kw_only=True)
    locked_cols: list[str] = field(default_factory=list, kw_only=True)
    no_null_cols: list[str] = field(default_factory=list, kw_only=True)
    target_cols: list[str] = field(default_factory=list, kw_only=True)
    date_cols: list[str] = field(default_factory=list, kw_only=True)
    index_col: str = field(default="ID", kw_only=True)

    dataframe: pl.DataFrame = field(init=False)

    def __post_init__(self, data_paths: Iterable[Path | str], separator: str) -> None:
        """Initialize the data frame with the data paths."""
        types = {}
        if self.index_col is not None:
            types[self.index_col] = pl.String

        dataframes = [
            pl.read_csv(
                data_path,
                ignore_errors=True,
                separator=separator,
                schema_overrides=types,
            )
            for data_path in data_paths
        ]
        self.dataframe = pl.concat(dataframes, how="vertical_relaxed")

        if not self.index_col or self.index_col not in self.dataframe.columns:
            ids = [uuid4() for i in range(len(self.dataframe))]
            self.with_columns(pl.Series(self.index_col, ids))

    @property
    def info_cols(self) -> list[str]:
        """Get the columns that must not be processed"""
        return [
            self.index_col,
            *self.hidden_cols,
            *self.target_cols,
        ]

    def describe(
        self,
        percetiles: Sequence[float] | float | None = None,
        show: bool = True,
    ) -> pl.DataFrame:
        """Get information about the data frame."""
        data = self.dataframe.describe(percentiles=percetiles)

        if show:
            display(data)

        return data

    def corr(
        self,
        with_nulls: bool = True,
        show: bool = True,
        size: float = 0.9,
        dpi=9,
        sub_sample: float = 1.0,
    ) -> pl.DataFrame:
        """Get the correlation matrix of the data frame."""
        temp_df = self.dataframe.drop(self.index_col).sample(fraction=sub_sample)
        if pl.String in temp_df.dtypes:
            raise ValueError("Cannot calculate correlation with String columns")

        temp_df = temp_df.clone()
        display(temp_df)

        if not with_nulls:
            temp_df = temp_df.drop_nulls()

        corr_data = (
            temp_df.select(
                *self.target_cols,
                cs.exclude(self.target_cols),
            )
            .to_dummies(cs.categorical())
            .corr()
        )

        if show:
            plt.figure(figsize=(len(temp_df.columns) * size, dpi))
            sns.heatmap(
                corr_data,
                annot=True,
                xticklabels=corr_data.columns,
                yticklabels=corr_data.columns,
            )
            plt.show()

        return corr_data

    def get_columns_types(self) -> dict[type[pl.DataType] | pl.Categorical, list[str]]:
        column_types = {}
        for col, dtype in zip(self.dataframe.columns, self.dataframe.dtypes):
            if col == self.index_col:
                continue
            if dtype not in column_types:
                column_types[dtype] = []
            column_types[dtype].append(col)

        print("Data Types:", {key: len(value) for key, value in column_types.items()})

        return column_types

    def clean_nulls(self, thresshold: float = 0.5, show: Show = Show.TITLE) -> Self:
        """Clean null values from the data frame."""
        print("Original Null Percentage\n----------------------")

        self.null_percent(show=show)

        print("----------------------\n")

        print("After Dropping Rows with Null Values on NO_NULL_COLUMNS:\n----------------------")

        no_null_cols = [col for col in self.no_null_cols if col in self.dataframe.columns]
        print(no_null_cols)
        if no_null_cols:
            self.dataframe = self.dataframe.drop_nulls(no_null_cols)

        new_nulls = self.null_percent(show=show)

        print("----------------------\n")

        print(
            f"Columns to Drop due to Null Percentage | {thresshold=:.0%} |\n----------------------"
        )

        columns_to_drop: list[str] = new_nulls.filter(pl.col("Null_Percentage") > thresshold * 100)[
            "Features"
        ].to_list()

        non_dropeable_columns = [*self.locked_cols, *self.no_null_cols, *self.target_cols]
        columns_to_drop = [col for col in columns_to_drop if col not in non_dropeable_columns]
        print(columns_to_drop)

        for drop_col in columns_to_drop:
            self.dataframe.drop_in_place(drop_col)
        self.null_percent(show=show)

        print("----------------------\n")

        return self

    def null_percent(
        self,
        show: Show = Show.PLOT,
        threshold: float | None = None,
    ) -> pl.DataFrame:
        """Get the percentage of null values in the data frame."""
        data: pl.DataFrame = self.dataframe.null_count() / len(self.dataframe)

        if threshold is not None:
            data = data.filter(data > threshold)

        data *= 100
        data = data.transpose(
            include_header=True,
            header_name="Features",
            column_names=["Null_Percentage"],
        ).sort("Null_Percentage")

        if show != Show.NONE:
            nulls = self.dataframe.null_count().sum_horizontal().sum()
            rows_with_nulls = len(self.dataframe.filter(pl.any_horizontal(pl.all().is_null())))
            nulls_per_row = nulls / rows_with_nulls if rows_with_nulls > 0 else 0
            print(
                "| "
                f"Total Rows: {len(self.dataframe):,} | "
                f"Rows with Nulls: {rows_with_nulls:,} | "
                f"Total Nulls: {nulls:,} | "
                f"Nulls per Row: {nulls_per_row:.2f} | "
            )

        match show:
            case Show.PLOT:
                display(data.plot.bar(x="Features", y="Null_Percentage"))
            case Show.DATA:
                display(data)
            case _:  # Show.NONE
                pass

        return data

    def column_distribution(
        self,
        column: Sequence[str] | str,
        show: Show = Show.PLOT,
        strict: bool = False,
    ) -> list[pl.Series]:
        """Get the distribution of a column."""
        if isinstance(column, str):
            column = [column]

        if strict:
            column = [col for col in column if col in self.dataframe.columns]

        data = []
        for col in column:
            item = self[col]
            data.append(item)

            match show:
                case Show.DATA:
                    display(item)

                case Show.PLOT:
                    length = item.count()
                    item = item.value_counts()
                    item = item.with_columns((pl.col("count") / length) * 100)
                    display(item.plot.bar(x=col, y="count", text="count"))

                case _:  # Show.NONE
                    pass

        return data

    def with_columns(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        inplace: bool = True,
        **named_exprs: IntoExpr,
    ) -> pl.DataFrame:
        """Add/Modify columns to the data frame.

        See Documentation on how to use this method on the Polars documentation.
        """
        dataframe = self.dataframe.with_columns(*exprs, *named_exprs)
        if inplace:
            self.dataframe = dataframe

        return dataframe

    def __getitem__(self, key: str) -> pl.Series:
        """Get a column from the data frame."""
        return self.dataframe[key]

    def __setitem__(self, key: str, value: pl.Series) -> None:
        """Set a column in the data frame."""
        self.dataframe[key] = value

    def __len__(self) -> int:
        """Get the length of the data frame."""
        return len(self.dataframe)


