import matplotlib.pyplot as plt


def plot_numeric_distribution(df, column):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(df[column].dropna())
    ax.set_title(f"{column} Distribution")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")

    return fig


def plot_categorical_numeric(df, categorical_column, numeric_column):
    fig, ax = plt.subplots(figsize=(8, 5))

    values = df.groupby(categorical_column)[numeric_column].mean()
    values.plot(kind="bar", ax=ax)

    ax.set_title(
        f"Average {numeric_column} by {categorical_column}"
    )
    ax.set_xlabel(categorical_column)
    ax.set_ylabel(f"Average {numeric_column}")
    ax.tick_params(axis="x", rotation=0)

    return fig

def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include="number")

    fig, ax = plt.subplots(figsize=(8, 6))

    correlation = numeric_df.corr()

    image = ax.imshow(correlation, cmap="coolwarm")
    fig.colorbar(image, ax=ax)

    ax.set_xticks(range(len(numeric_df.columns)))
    ax.set_xticklabels(numeric_df.columns, rotation=45)

    ax.set_yticks(range(len(numeric_df.columns)))
    ax.set_yticklabels(numeric_df.columns)

    ax.set_title("Correlation Heatmap")

    fig.tight_layout()

    return fig