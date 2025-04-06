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
    print(df.glimpse())
    return csv_file, df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
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
def _(df, pl):
    (
        df
        .select(
            # Identity expression
            pl.col("Pclass"),
            # Names to lowercase
            pl.col("Name").str.to_lowercase(),
            # Round the ages
            pl.col("Age").round(2)
        )
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## @TODO split name to 3 parts and pclas set as categorical""")
    return


if __name__ == "__main__":
    app.run()
