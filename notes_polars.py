import marimo

__generated_with = "0.12.4"
app = marimo.App(width="medium", app_title="Polars notes")


@app.cell
def _(mo):
    mo.md(
        r"""
        # Polars notes

        Based on course created by Liam Brannigan: https://www.udemy.com/course/data-analysis-with-polars/
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import polars as pl
    return (pl,)


@app.cell
def _():
    import numpy as np
    return (np,)


@app.cell
def _(pl):
    csv_file = "./data/titanic.csv"
    df = pl.read_csv("./data/titanic.csv")
    return csv_file, df


@app.cell
def _(df):
    df.glimpse()
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
    mo.md(r"""### Grouping and aggregation""")
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
    mo.md(
        r"""
        ## Lazy mode
        * all previous examples works in eager mode
        * read csv in lazy mode by replacing `read_csv` with `scan_csv`
        * to see the optimized plan run
          ```python
          LazyFrame.explain(optimized=True)
          ```
        * to evaluate run
          ```python
          LazyFrame.collect()
          ```
        """
    )
    return


@app.cell
def _(csv_file, pl):
    ldf =(
        pl.scan_csv(csv_file)
        .group_by("Survived","Pclass")
        .agg(
            pl.col("PassengerId").count().alias("counts")
        )
    )
    print(ldf.explain(optimized=True))
    ldf.collect()
    return (ldf,)


@app.cell
def _(mo):
    mo.md(
        r"""
        When polars opens csv in lazy mode it:

        * opens the file 
        * gets the column names as headers
        * infers the type of each column from the first 100 rows
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        to check what types were inffered
          ```python
          ldf.schema
          ```
        `ldf.collect_schema()` does not process your data, it just runs through the optimised query plan
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        to create lazy frame from data:
        ```python
        pl.LazyFrame({"values":[0,1,2]})
        ```

        to change df into ldf:
        ```python
        ldf = df.lazy()
        ```
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""**methods called on a `DataFrame` acts on the data. An method on a `LazyFrame` acts on the query plan**""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Streaming larger-than-memory datasets
        * when streaming is enabled ploars process dataframe in chunks
        * To enable streaming pass to `collect` argument `streaming = True`
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Optimizations applied by Polars include:

        - `projection pushdown` limit the number of columns read to those required
        - `predicate pushdown` apply filter conditions as early as possible
        - `combine predicates` combine multiple filter conditions
        - `slice pushdown` limit rows processed when limited rows are required
        - `common subplan elimination` run duplicated transformations on the same data once and then re-use
        - `common subexpression elimination` duplicated expressions are cached and re-used
        """
    )
    return


app._unparsable_cell(
    r"""
    Partial evaluation
    ```python
    (
        df
        .head(3)
        .collect()
    )
    ```
    """,
    name="_"
)


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Excercise 1.2.1
        Create a `LazyFrame` by doing a scan of the Titanic CSV file

        Check to see which of the following metadata you can get from a `LazyFrame`:

        - number of rows
        - column names
        - schema

        Create a lazy query where you scan the Titanic CSV file and then select the `Name` and `Age` columns.

        Print out the optimised query plan for this query
        """
    )
    return


@app.cell
def _(csv_file, df, pl):
    exc_ldf = pl.scan_csv(csv_file)
    # exc_ldf.shape return error
    print(df.columns)
    print(df.schema)

    (
        pl.scan_csv(csv_file)
        .select("Name","Age")
        .explain()
    )
    return (exc_ldf,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Streaming larger-than-memory datasets

        * when streaming is enabled ploars process dataframe in chunks
        * To enable streaming pass to `collect` argument `streaming=True`
        * Streaming is not supported for all operations. However, many key operations such as `filter`, `groupby` and `join` support streaming. If streaming is not possible then Polars will run the query without streaming.
        """
    )
    return


@app.cell
def _(csv_file, pl):
    (
        pl.scan_csv(csv_file)
        .filter(pl.col("Age") > 50)
        .group_by(["Survived","Pclass"])
        .agg(
            pl.col("PassengerId").count().alias("counts")
        )
        .collect(engine="streaming")
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Limits of lazy mode

        There are operations that cannot be done in lazy mode.
        In this case it's recommended to stay in lazy mode as long as posible. Convert to eager data frame for problematic operation and then conver back to lazy dataframe
        ```python
        (
            pl.scan_csv(csv_file)
            .collect()
            .pivot(index="Pclass",on="Sex",values="Age",aggregate_function="mean")
            .lazy()
        )
        ```
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Exercise 1.3.1
        Create a `LazyFrame` and covert it to eager mode

        ```python
        (
            pl.scan_csv(csv_file)
            .head()
            .collect()
            .lazy()
            .head(3)
        )
        ```
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Data frame methods

        * df.width
        * df.height
        * df.schema
        * df.dtypes

        Metod for comparing shemas

        ```python
        def compare_polars_schema(
                df_schema:pl.Schema, 
                target_schema: pl.Schema
        ):
        '''
        Compare two pl.Schema and report on any differences
        Args:
            df_schema (OrderedDict): The schema of our DataFrame
            target_schema (OrderedDict): The target schema of our DataFrame that we are comparing to
    
        Returns:
            Dict containing comparison details, with keys indicating the type of difference
        '''
            # Check if they are the same
            if df_schema == target_schema:
                return {"match": True}
    
            # Otherwise do a detailed comparison
            comparison_result = {
                "match": False,
                "differences": {}
            }
    
            # Check keys
            df_keys = set(df_schema.keys())
            target_keys = set(target_schema.keys())
    
            # Check for missing or extra keys
            missing_in_target_schema = df_keys - target_keys
            missing_in_df_schema = target_keys - df_keys
    
            if missing_in_target_schema:
                comparison_result["differences"]["keys_missing_in_target"] = list(missing_in_target_schema)
    
            if missing_in_df_schema:
                comparison_result["differences"]["keys_missing_in_df"] = list(missing_in_df_schema)
    
            # Check common keys for dtype differences
            common_keys = df_keys.intersection(target_keys)
    
            dtype_differences = {}
            for key in common_keys:
                if df_schema[key] != target_schema[key]:
                    dtype_differences[key] = {
                        "df_type": str(df_schema[key]),
                        "target_type": str(target_schema[key])
                    }
    
            if dtype_differences:
                comparison_result["differences"]["dtype_mismatches"] = dtype_differences
    
            return comparison_result
        ```
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r""" """)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        polars use for cols apache arrow

        ```python
        df["Age"].to_arrow()
        ```

        A Polars `DataFrame` holds references to an Arrow Table which holds references to Arrow Arrays. We can think of a Polars `DataFrame` being a lightweight object that points to the lightweight Arrow Table which points to the heavyweight Arrow Arrays (heavyweight because they hold the actual data). 


        """
    )
    return


@app.cell
def _(np, pl):
    # @TODO
    df_shape = (1_000_000,100)
    df_polars = pl.DataFrame(
        np.random.standard_normal(df_shape)
    )
    df_polars.head(3)
    return df_polars, df_shape


if __name__ == "__main__":
    app.run()
