import marimo

__generated_with = "0.14.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from marimo_scipy_utils import (
        display_sliders,
        generate_ranges,
        params_sliders,
        sample_invars,
    )

    return (
        display_sliders,
        generate_ranges,
        mo,
        np,
        params_sliders,
        plt,
        sample_invars,
    )


@app.cell
def _(mo):
    mo.md(
        """
        # marimo-scipy-utils demo

        Configure input variables as distributions (or constants), then run a
        simple Monte Carlo simulation with `sample_invars`.
        """
    )
    return


@app.cell
def _():
    invars = {}
    return (invars,)


@app.cell
def _(generate_ranges, params_sliders):
    revenue_sliders = params_sliders(
        generate_ranges(
            "norm",
            {
                "loc": {"lower": 100_000, "upper": 1_000_000, "step": 10_000},
                "scale": {"upper": 200_000, "step": 1_000, "value": 50_000},
            },
        )
    )
    return (revenue_sliders,)


@app.cell
def _(display_sliders, invars, revenue_sliders):
    revenue_view = display_sliders(
        "Annual revenue", revenue_sliders, invars, dist="norm"
    )
    revenue_view
    return (revenue_view,)


@app.cell
def _(mo):
    margin_slider = mo.ui.slider(start=0.0, stop=1.0, step=0.01, value=0.3)
    return (margin_slider,)


@app.cell
def _(display_sliders, invars, margin_slider):
    margin_view = display_sliders("Profit margin", margin_slider, invars)
    margin_view
    return (margin_view,)


@app.cell
def _(invars, margin_view, np, revenue_view, sample_invars):
    _ = (revenue_view, margin_view)  # make sampling depend on slider registration
    samples = sample_invars(invars, 10_000, rng=np.random.default_rng(0))
    profit = samples["Annual revenue"] * samples["Profit margin"]
    return (profit,)


@app.cell
def _(mo, plt, profit):
    from marimo_scipy_utils import abbrev_format

    _fig, _ax = plt.subplots(figsize=(6, 3))
    _ax.hist(profit, bins=50, color="C0", alpha=0.7)
    _ax.set_xlabel("Simulated annual profit")
    _ax.set_ylabel("Count")
    _ax.xaxis.set_major_formatter(plt.FuncFormatter(abbrev_format))
    _chart = mo.as_html(_fig)
    plt.close(_fig)
    mo.vstack(
        [
            mo.md(
                f"### Simulated profit (10,000 draws) — "
                f"median {abbrev_format(float(sorted(profit)[len(profit) // 2]))}"
            ),
            _chart,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
