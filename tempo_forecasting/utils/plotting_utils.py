import pandas as pd

import matplotlib.pyplot as plt
from typing import Sequence, Union, Optional
from types import NoneType

def core_plotter(title: str,
                ax,
                x_label: str, 
                y_label: str,
                actual_dates: Sequence[pd.Timestamp], 
                actual_vals: Sequence[Union[int, float]],
                fitted_train_dates: Optional[Sequence[pd.Timestamp]] = [], 
                fitted_train_vals: Optional[Sequence[Union[int, float]]] = [],
                forecasted_test_dates: Optional[Sequence[pd.Timestamp]] = [], 
                forecasted_test_vals: Optional[Sequence[Union[int, float]]] = [],
                forecasted_holdout_dates: Optional[Sequence[pd.Timestamp]] = [], 
                forecasted_holdout_vals: Optional[Sequence[Union[int, float]]] = [],
                forecasted_dates: Optional[Sequence[pd.Timestamp]] = [], 
                forecasted_vals: Optional[Sequence[Union[int, float]]] = [],
                lw: Optional[float] = 1.0):
    """
    Plots time series demand data with the option to include any of several other related data series:
      fitted (train period), forecasted (test period), forecasted (hold out period) and general forecast.

    Parameters:
        title (str): The title of the plot.
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.
        actual_dates (sequence): A list of dates corresponding to the actual demand dates.
        actual_vals (sequence): A list of actual demand values.
        fitted_train_dates (sequence, optional): A list of dates for the fitted training values. Default is an empty list.
        fitted_train_vals (sequence, optional): A list of fitted training values to be plotted. Default is an empty list.
        forecasted_test_dates (sequence, optional): A list of dates for forecasted test values. Default is an empty list.
        forecasted_test_vals (sequence, optional): A list of forecasted test values to be plotted. Default is an empty list.
        forecasted_holdout_dates (sequence, optional): A list of dates for forecasted holdout values. Default is an empty list.
        forecasted_holdout_vals (sequence, optional): A list of forecasted holdout values to be plotted. Default is an empty list.
        forecasted_dates (sequence, optional): A list of dates for general forecasted values. Default is an empty list.
        forecasted_vals (sequence, optional): A list of forecasted values to be plotted. Default is an empty list.
        lw (float, optional): A float indicating line width. Defaults to 1.

    Returns:
        None: This function does not return any value; it displays the plot directly.

    """

    # color ref: https://matplotlib.org/stable/gallery/color/named_colors.html
    # navy, darkviolet, darkturquoise, orange works pretty well
    # forestgreen is alright

    ax.plot(actual_dates, actual_vals, label="Actual Demand", color = "lightsteelblue")
    if (len(fitted_train_dates) > 0) & (len(fitted_train_vals) >0):
        ax.plot(fitted_train_dates, fitted_train_vals, label="Fitted Demand", linestyle="--", linewidth=lw, color = "darkblue")
    if (len(forecasted_test_dates) >0) & (len(forecasted_test_vals) >0):
        ax.plot(forecasted_test_dates, forecasted_test_vals, label="Predicted Demand (testing)", linestyle="--", linewidth=lw, color = "magenta")
    if (len(forecasted_holdout_dates) >0) & (len(forecasted_holdout_vals) >0):
        ax.plot(forecasted_holdout_dates, forecasted_holdout_vals, label="Predicted Demand (final hold out)", linestyle="--", linewidth=lw, color = "orange")
    if (len(forecasted_dates) >0) & (len(forecasted_vals) >0):
        ax.plot(forecasted_dates, forecasted_vals, label="Forecasted Demand", linestyle="--", linewidth=lw, color = "limegreen")

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    ax.grid(True)


def plot_time_series_demand(title: str, 
                            x_label: str, 
                            y_label: str,
                            actual_dates: Sequence[pd.Timestamp], 
                            actual_vals: Sequence[Union[int, float]],
                            fitted_train_dates: Optional[Sequence[pd.Timestamp]] = [], 
                            fitted_train_vals: Optional[Sequence[Union[int, float]]] = [],
                            forecasted_test_dates: Optional[Sequence[pd.Timestamp]] = [], 
                            forecasted_test_vals: Optional[Sequence[Union[int, float]]] = [],
                            forecasted_holdout_dates: Optional[Sequence[pd.Timestamp]] = [], 
                            forecasted_holdout_vals: Optional[Sequence[Union[int, float]]] = [],
                            forecasted_dates: Optional[Sequence[pd.Timestamp]] = [], 
                            forecasted_vals: Optional[Sequence[Union[int, float]]] = [],
                            fig_width = 12,
                            fig_height = 6):
    
    fig, ax = plt.subplots()
    fig.set_figwidth(fig_width)
    fig.set_figheight(fig_height)

    core_plotter(title = title,
                 ax = ax,
                 x_label = x_label, 
                 y_label = y_label,
                 actual_dates = actual_dates, 
                 actual_vals = actual_vals,
                 fitted_train_dates = fitted_train_dates, 
                 fitted_train_vals = fitted_train_vals,
                 forecasted_test_dates = forecasted_test_dates, 
                 forecasted_test_vals = forecasted_test_vals,
                 forecasted_holdout_dates = forecasted_holdout_dates,
                 forecasted_holdout_vals = forecasted_holdout_vals,
                 forecasted_dates = forecasted_dates,
                 forecasted_vals = forecasted_vals)

def plot_train_and_pred(title: str, 
                    x_label: str, 
                    y_label: str,
                    actual_dates: Sequence[pd.Timestamp], 
                    actual_vals: Sequence[Union[int, float]],
                    fitted_train_dates: Optional[Sequence[pd.Timestamp]] = [], 
                    fitted_train_vals: Optional[Sequence[Union[int, float]]] = [],
                    forecasted_test_dates: Optional[Sequence[pd.Timestamp]] = [], 
                    forecasted_test_vals: Optional[Sequence[Union[int, float]]] = [],
                    refitted_dates: Optional[Sequence[pd.Timestamp]] = [], 
                    refitted_vals: Optional[Sequence[Union[int, float]]] = [],
                    forecasted_dates: Optional[Sequence[pd.Timestamp]] = [], 
                    forecasted_vals: Optional[Sequence[Union[int, float]]] = [],
                    caption: Optional[str] = None):
    
    fig, (ax1, ax2) = plt.subplots(1,2, sharey=True)
    fig.set_figheight(3)
    fig.set_figwidth(16)
    fig.subplots_adjust(top=0.8)
    fig.suptitle(title)
    if caption:
        caption_lines = caption.count("\n") + 1
        if caption_lines == 1:
            fig.text(0.1, -0.1, caption)
        else:
            fig.text(0.1, -0.075*caption_lines, caption)
    
    core_plotter("Training",
                 ax = ax1,
                 x_label = x_label, 
                 y_label = y_label,
                 actual_dates = actual_dates, 
                 actual_vals = actual_vals,
                 fitted_train_dates = fitted_train_dates, 
                 fitted_train_vals = fitted_train_vals,
                 forecasted_test_dates = forecasted_test_dates, 
                 forecasted_test_vals = forecasted_test_vals,
                 lw = 0.75)
    
    core_plotter("Forecasting",
                 ax = ax2,
                 x_label = x_label, 
                 y_label = y_label,
                 actual_dates = actual_dates, 
                 actual_vals = actual_vals,
                 fitted_train_dates = refitted_dates, 
                 fitted_train_vals = refitted_vals,
                 forecasted_dates = forecasted_dates, 
                 forecasted_vals = forecasted_vals,
                 lw = 0.75)

# POST-MODELING CHECKS
def extract_series_from_date_vals(category_date_vals: pd.DataFrame, 
                                  rename_date_col: str = "date", 
                                  rename_val_col: str = "vals"):
    """
    Processes a DataFrame containing categorical date values and extracts various 
    time series segments based on specified date conditions such as actuals, 
    fitting, and forecast. The function organizes these segments into a dictionary, 
    renaming the columns according to provided arguments for better usability.

    Parameters:
        cat_vals_pd (pd.DataFrame): A pandas DataFrame containing date and value columns 
                                    relevant for the time series category. Expected columns: 
                                    "date", "true_vals", "eval_train_preds", "eval_test_preds", 
                                    "eval_train_preds", "final_train_preds", "forecast.
        rename_date_col (str): The name to which the date column should be renamed in 
                                the output dictionary.
        rename_val_col (str): The name to which the value columns should be renamed in 
                            the output dictionary. This applies to "true_vals", 
                            "full_train_refit", "test_pred", and "forecast".

    Returns:
        dict: A dictionary containing the processed time series segments with keys:
            - "actuals_full" (pd.DataFrame): DataFrame of all actual values up to 
            the maximum actual date.
            - "actuals_train" (pd.DataFrame): DataFrame containing actual values in 
            the initial training period.
            - "fitted_train" (pd.DataFrame): DataFrame containing fitted values from 
            the full training refit during the initial training period.
            - "actuals_test" (pd.DataFrame): DataFrame of actual values during the 
            initial testing period.
            - "forecasted_test" (pd.DataFrame): DataFrame of forecasted values during 
            the initial testing period.
            - "refitted_train" (pd.DataFrame): DataFrame with refitted training values.
            - "forecasted" (pd.DataFrame): DataFrame containing forecasted values beyond 
            the maximum actual date.    
    """

    cat_vals_pd = category_date_vals.sort_values("date",ascending=True)
    cat_vals_pd["date"] = pd.to_datetime(cat_vals_pd["date"]).dt.date

    # Define Individual Series
    actuals_full = cat_vals_pd[cat_vals_pd["true_vals"].notnull()][["date","true_vals"]]
    actuals_full.rename(columns={'date': rename_date_col, 'true_vals': rename_val_col}, inplace=True)

    actuals_train = cat_vals_pd[cat_vals_pd["eval_train_preds"].notnull()][["date","true_vals"]]
    actuals_train.rename(columns={'date': rename_date_col, 'true_vals': rename_val_col}, inplace=True)

    actuals_test = cat_vals_pd[cat_vals_pd["eval_test_preds"].notnull()][["date","true_vals"]]
    actuals_test.rename(columns={'date': rename_date_col, 'true_vals': rename_val_col}, inplace=True)

    fitted_train = cat_vals_pd[cat_vals_pd["eval_train_preds"].notnull()][["date","eval_train_preds"]]
    fitted_train.rename(columns={'date': rename_date_col, 'eval_train_preds': rename_val_col}, inplace=True)

    forecasted_test = cat_vals_pd[cat_vals_pd["eval_test_preds"].notnull()][["date","eval_test_preds"]]
    forecasted_test.rename(columns={'date': rename_date_col, 'eval_test_preds': rename_val_col}, inplace=True)

    refitted_train = cat_vals_pd[cat_vals_pd["final_train_preds"].notnull()][["date","final_train_preds"]]
    refitted_train.rename(columns={'date': rename_date_col, 'final_train_preds': rename_val_col}, inplace=True)

    forecasted = cat_vals_pd[cat_vals_pd["forecast"].notnull()][["date","forecast"]]
    forecasted.rename(columns={'date': rename_date_col, 'forecast': rename_val_col}, inplace=True)

    # make it usable
    series_dict = {
        "actuals_full": actuals_full,
        "actuals_train": actuals_train,
        "fitted_train": fitted_train,
        "actuals_test": actuals_test,
        "forecasted_test": forecasted_test,
        "refitted_train": refitted_train,
        "forecasted": forecasted
    }

    return series_dict

def plot_side_by_side(category: str, 
                      all_date_vals: pd.DataFrame, 
                      final_params: pd.DataFrame = None, 
                      caption_add = "",
                      title_add = "") -> None:
    """
    Plots the training actual/fitted/test prediction values, 
    and the forecasting actual/fitted/forecasted values
    for a specified category in side by side plots.

    This function extracts the relevant time series data for a given category and generates a plot 
    that displays the actual machine counts, fitted values from the training data, and forecasted 
    values. The plot includes both actual and forecasted dates labeled appropriately.

    Parameters:
        category (str): The category for which to plot the time series data.
        all_date_vals (DataFrame): A DataFrame containing date values and associated metrics
                                    for all categories.
        final_params (DataFrame): A DataFrame containing the final model parameters for each 
                                category, including the model instance names.
        caption_add (str, optional): Additional text to include in the plot caption. 
                                    Defaults to an empty string.
        title_add (str, optional): Additional text to include in the plot's title.
                                    Defaults to an empty string.

    Returns:
        None: This function generates a plot but does not return any values.
    """

    cat_all_date_vals = all_date_vals[all_date_vals["category"] == category]
    data_series_dict = extract_series_from_date_vals(cat_all_date_vals, 
                                                    rename_date_col = "date", 
                                                    rename_val_col = "vals")

    # Plot it
    if type(final_params) != NoneType:
        cat_final_params = final_params[final_params["category"] == category]
        best_model = cat_final_params["model_name"][0]

        caption = f"{best_model} model" + caption_add
    else:
        caption = caption_add

    plot_train_and_pred(title = f"Category: {category}{title_add}", 
                        x_label = "Date", 
                        y_label = "Demand",
                        actual_dates = data_series_dict["actuals_full"]["date"], 
                        actual_vals = data_series_dict["actuals_full"]["vals"],
                        fitted_train_dates = data_series_dict["fitted_train"]["date"], 
                        fitted_train_vals = data_series_dict["fitted_train"]["vals"],
                        forecasted_test_dates = data_series_dict["forecasted_test"]["date"], 
                        forecasted_test_vals = data_series_dict["forecasted_test"]["vals"],
                        refitted_dates = data_series_dict["refitted_train"]["date"], 
                        refitted_vals = data_series_dict["refitted_train"]["vals"],
                        forecasted_dates = data_series_dict["forecasted"]["date"], 
                        forecasted_vals = data_series_dict["forecasted"]["vals"],
                        caption = caption)