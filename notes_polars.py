import marimo

__generated_with = "0.12.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import polars as pl
    return (pl,)


@app.cell
def _(pl):
    csv_file = "/data/titanic.csv"
    df = pl.read_csv("./data/titanic.csv")
    return csv_file, df


@app.cell
def _(df):
    print(df.glimpse())
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Expression API
        * using expresion API instead of square brackets let polars do ``parallelisation`` and ``query optimisation``
        * in the Expression API we use `pl.col` to refer to a column
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### Select""")
    return


@app.cell
def _(df, pl):
    (
        df
        .select(
            # Identity expression
            pl.col("Pclass").cast(pl.Utf8).cast(pl.Categorical),
            # Name parsing
            pl.col("Name").str.extract_groups(
                r'^(?P<surname>\w+),\s(?P<title>\w+\.)\s(?P<given_name>.*?)(?:\s\(.*\))?$'
            ).alias("name_parts"),
            # Round the ages
            pl.col("Age").round(2)
        )
        .unnest("name_parts")
        .select([
            "Pclass",        # Rest of the columns
            "Age",
            "title",         # Now comes FIRST
            "surname",       # Then surname
            "given_name",    # Then given name
        ])
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### Filter""")
    return


@app.cell
def _(df, pl):
    (
        df
        .filter(
            (pl.col("Age") > 70) &
            (pl.col("Pclass") == 3)
        )
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### Group by""")
    return


@app.cell
def _(df, pl):
    (
        df
        .group_by(["Survived","Pclass"])
        .agg(
            pl.col("PassengerId").count().alias("counts")
        )
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Visualisation

        * polars also has a built-in `plot` methods that allow us to create an Altair chart directly from a dataframe
        * `pip install altair`
        * altair process dataframe as whole and sometimies can thor error "The number of rows in your dataset is greater than the maximum allowed (5000)." Solutions
            * Increase the limit (for exploration):
            ```python
            import altair as alt
            alt.data_transformers.enable('default', max_rows=None)  # Disable limit
            ```
            * Sample your data (recommended for large datasets):
            ```python
            chart = alt.Chart(df.sample(1000))  # Plot random subset
            ```
            * Use data aggregation (best for performance):
            ```python
            chart = alt.Chart(df).transform_aggregate(
                count='count()',
                groupby=['column_of_interest']
            ).mark_bar()
            ```
            * Enable Vega-Lite's data server (for big data):
            ```python
            alt.data_transformers.enable('json')
            ```
        """
    )
    return


@app.cell
def _(df):
    (
        df
        .plot
        .scatter(
            x="Age",
            y="Fare"
        )
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Lazy mode""")
    return


if __name__ == "__main__":
    app.run()
