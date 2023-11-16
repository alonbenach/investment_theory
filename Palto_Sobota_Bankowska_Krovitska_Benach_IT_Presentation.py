# %%
# Import all required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
import matplotlib.dates as mdates
from fredapi import Fred

# %%
### Load data:
# Ticker symbol for the S&P GSCI Energy Index
ticker_symbol = "^SPGSCI"

# Download data for the S&P GSCI Energy Index
sp_gsci_energy = yf.download(ticker_symbol, start="2000-12-01", end="2022-12-31")

print(sp_gsci_energy.head())
# %%
### Load risk free rate data
f = "F-F_Research_Data_Factors.CSV"
rf_mkt = pd.read_csv(f, parse_dates=["Date"], index_col="Date")

# %%
#################################Stage 1: Analysis of SPGSCI Index####################################
### Compute monthly returns:
sp_gsci_energy["Year_Month"] = sp_gsci_energy.index.to_period("M")

# Calculate monthly returns
monthly_returns = (
    sp_gsci_energy["Adj Close"].resample("M").ffill().pct_change().dropna()
)

print(monthly_returns)
# %%
### Calculate basic statistics:
# Calculate mean and standard deviation
stats = monthly_returns.describe().loc[["mean", "std"]]

print(stats)

# %%
### Calculate excess returns
# Resample ' RF ' column to end-of-month frequency
rf_adjusted = rf_mkt[" RF "].resample("M").last()

# Subtract 'RF' column from monthly_returns
excess_returns = monthly_returns - rf_adjusted

print(excess_returns)
# Calculate average excess return and tandard deviation:
excess_returns_stats = excess_returns.describe().loc[["mean", "std"]]
print(excess_returns_stats)


# %%
print(
    "Basic statistics for monthly returns on the index: \n",
    f"Mean: {(stats[0]*100):.3f}% \n Standard Deviation: {(stats[1])*100:.3f}%",
)
print(
    "Basic statistics for monthly excess returns on the index: \n",
    f"Mean: {(excess_returns_stats[0])*100:.3f}% \n Standard Deviation: {(excess_returns_stats[1]*100):.3f}%",
)

# %%
sharpe_ratio = excess_returns_stats["mean"] / stats["std"]
print(
    "Sharpe Ratio for the time series of excess return of the index: \n",
    f"{sharpe_ratio:.3f}%",
)

# %%
# Calculate annualized average return
annualized_mean_return = monthly_returns.mean() * 12
# Calculate annualized standard deviation of return
annualized_std_excess_return = monthly_returns.std() * (12**0.5)
# Calculate annualized excess return
annualized_excess_return = excess_returns.mean() * 12
# Calculate annualized standard deviation of excess returns
annualized_std_excess_return = excess_returns.std() * (12**0.5)
# Calculate annualized Sharpe Ratio
annualized_sharpe_ratio = annualized_excess_return / annualized_std_excess_return

# Create a Series with the calculated statistics
annualized_stats = pd.Series(
    {
        "Annualized Average Return": f"{(annualized_mean_return)*100:.3f}%",
        "Annualized Excess Return": f"{(annualized_excess_return)*100:.3f}%",
        "Annualized Sharpe Ratio": f"{annualized_sharpe_ratio:.3f}%",
    }
)
print(annualized_stats)

# %%
### Time series of investment of $1 into the commodities index to compute its development over time
investment_development = (1 + monthly_returns).cumprod()

print(investment_development)
# %%
### Get CPI data to construct an inflation rate series
fred = Fred(api_key="508678eb194acfbaf6a3295ecd5e5c35")

# Fetch US inflation data (Consumer Price Index)
us_inflation = fred.get_series(
    "CPIAUCNS", start_date="2001-01-01", end_date="2022-12-31"
)

# Trimming the data to the required date range
us_cpi_trimmed = us_inflation.loc["2000-12-01":"2022-12-31"]

# Calculating percentage change in CPI
inflation_rate = us_cpi_trimmed.pct_change() * 100
inflation_rate = inflation_rate.dropna()

inflation_rate.index = inflation_rate.index + pd.tseries.offsets.MonthEnd()
print(inflation_rate.head())


# %%
# Visualize the time series of $1 investment
# Convert the index to datetime if not already in datetime format
investment_development.index = pd.to_datetime(investment_development.index)

# Create a figure and plot the data with a dark background and grid pattern
plt.figure(figsize=(12, 6))

# Setting dark background with grid pattern
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.lineplot(data=investment_development, color="blue", linewidth=2)

# Show only the calendar year in the x-axis ticks with 45-degree rotation
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.xticks(rotation=45, color="white")  # Setting x-ticks color to white

# Customizing spines and gridlines
sns.despine(left=True, bottom=True)
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.title("Development of $1 Investment in SPGSCI Index", fontsize=16, color="white")
plt.xlabel("Date", fontsize=12, color="white")
plt.ylabel("Value of Investment", fontsize=12, color="white")

# Background color customization
plt.gca().set_facecolor("#303030")
plt.gcf().patch.set_facecolor("#303030")

# Set y-ticks color to white
plt.gca().tick_params(axis="y", colors="white")

plt.show()

# %%
# Visualize the time series of $1 investment with inflation trend
# Convert the index to datetime if not already in datetime format
investment_development.index = pd.to_datetime(investment_development.index)
us_inflation.index = pd.to_datetime(us_inflation.index)

# Create a figure and plot the data with a dark background and grid pattern
plt.figure(figsize=(12, 6))

# Setting dark background with grid pattern
sns.set_style("darkgrid", {"axes.facecolor": ".9"})

# Plotting investment development
sns.lineplot(
    data=investment_development, color="blue", linewidth=2, label="SPGSCI Index"
)

# Plotting US inflation trend as a second line
sns.lineplot(data=inflation_rate, color="red", linewidth=2, label="US Inflation Rate")

# Show only the calendar year in the x-axis ticks with 45-degree rotation
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.xticks(rotation=45, color="white")  # Setting x-ticks color to white

# Customizing spines and gridlines
sns.despine(left=True, bottom=True)
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.title(
    "Development of $1 Investment and US Inflation (SPGSCI Index vs. US Inflation)",
    fontsize=16,
    color="white",
)
plt.xlabel("Date", fontsize=12, color="white")
plt.ylabel("Value", fontsize=12, color="white")

# Background color customization
plt.gca().set_facecolor("#303030")
plt.gcf().patch.set_facecolor("#303030")

# Set y-ticks color to white
plt.gca().tick_params(axis="y", colors="white")

# Marking distinct sub-periods with vertical lines
plt.axvline(pd.to_datetime("2005-12-31"), color="red", linestyle="-", alpha=0.5)
plt.axvline(pd.to_datetime("2010-12-31"), color="red", linestyle="-", alpha=0.5)
plt.axvline(pd.to_datetime("2015-12-31"), color="red", linestyle="-", alpha=0.5)

# Adding legend
plt.legend()

plt.show()

# %%
#################################Stage 2: Analysis of Sub-Periods####################################
### Visualise of planned subdivision:
# Create a figure and plot the data with a dark background and grid pattern
plt.figure(figsize=(12, 6))

# Setting dark background with grid pattern
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.lineplot(data=investment_development, color="blue", linewidth=2)

# Show only the calendar year in the x-axis ticks with 45-degree rotation
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.xticks(rotation=45, color="white")  # Setting x-ticks color to white

# Customizing spines and gridlines
sns.despine(left=True, bottom=True)
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.title(
    "Development of $1 Investment in SPGSCI Index - Divided into 4 Periods",
    fontsize=16,
    color="white",
)
plt.xlabel("Date", fontsize=12, color="white")
plt.ylabel("Value of Investment", fontsize=12, color="white")

# Background color customization
plt.gca().set_facecolor("#303030")
plt.gcf().patch.set_facecolor("#303030")

# Set y-ticks color to white
plt.gca().tick_params(axis="y", colors="white")

# Marking distinct sub-periods with vertical lines
plt.axvline(pd.to_datetime("2005-12-31"), color="red", linestyle="-", alpha=0.5)
plt.axvline(pd.to_datetime("2010-12-31"), color="red", linestyle="-", alpha=0.5)
plt.axvline(pd.to_datetime("2015-12-31"), color="red", linestyle="-", alpha=0.5)

plt.show()

# %%
# Divide the series into four sub-periods
returns_p1 = monthly_returns["2001-01-31":"2005-12-31"]
returns_p2 = monthly_returns["2006-01-31":"2010-12-31"]
returns_p3 = monthly_returns["2011-01-31":"2015-12-31"]
returns_p4 = monthly_returns["2016-01-31":"2022-12-31"]

# Calculate statistics for each sub-period
stats_period_1 = returns_p1.describe()[1:3]
stats_period_2 = returns_p2.describe()[1:3]
stats_period_3 = returns_p3.describe()[1:3]
stats_period_4 = returns_p4.describe()[1:3]

print("Statistics for 2001-2005 (returns):")
print(stats_period_1)
print("\nStatistics for 2006-2010 (returns):")
print(stats_period_2)
print("\nStatistics for 2011-2015 (returns):")
print(stats_period_3)
print("\nStatistics for 2016-2022 (returns):")
print(stats_period_4)

# %%
# Divide the series into four sub-periods
excess_returns_p1 = excess_returns["2001-01-31":"2005-12-31"]
excess_returns_p2 = excess_returns["2006-01-31":"2010-12-31"]
excess_returns_p3 = excess_returns["2011-01-31":"2015-12-31"]
excess_returns_p4 = excess_returns["2016-01-31":"2022-12-31"]

# Calculate statistics for each sub-period
stats_er_period_1 = excess_returns_p1.describe()[1:3]
stats_er_period_2 = excess_returns_p2.describe()[1:3]
stats_er_period_3 = excess_returns_p3.describe()[1:3]
stats_er_period_4 = excess_returns_p4.describe()[1:3]

print("Statistics for 2001-2005 (excess returns):")
print(stats_er_period_1)
print("\nStatistics for 2006-2010 (excess returns):")
print(stats_er_period_2)
print("\nStatistics for 2011-2015 (excess returns):")
print(stats_er_period_3)
print("\nStatistics for 2016-2022 (excess returns):")
print(stats_er_period_4)

# %%
# Calculate Sharpe ratio for each sub-period
sharpe_ratio_p1 = stats_er_period_1[0] / stats_period_1[1]
sharpe_ratio_p2 = stats_er_period_2[0] / stats_period_2[1]
sharpe_ratio_p3 = stats_er_period_3[0] / stats_period_3[1]
sharpe_ratio_p4 = stats_er_period_4[0] / stats_period_4[1]


# %%
stats_er_period_1.loc["sharpe_ratio"] = sharpe_ratio_p1
print(stats_er_period_1)
stats_er_period_2.loc["sharpe_ratio"] = sharpe_ratio_p2
print(stats_er_period_2)
stats_er_period_3.loc["sharpe_ratio"] = sharpe_ratio_p3
print(stats_er_period_3)
stats_er_period_4.loc["sharpe_ratio"] = sharpe_ratio_p4
print(stats_er_period_4)

# %%
### Visualize and compare the subperiods
# Regular returns:
# Mean:
# Extracting the 'mean' values from each Series
mean_values = [
    stats_period_1.loc["mean"],
    stats_period_2.loc["mean"],
    stats_period_3.loc["mean"],
    stats_period_4.loc["mean"],
]

# Creating labels for the x-axis (periods)
periods = ["Period 1", "Period 2", "Period 3", "Period 4"]

# Creating a bar plot for mean values
plt.figure(figsize=(8, 6))
bars = plt.bar(periods, mean_values, color="lightblue")
plt.xlabel("Periods")
plt.ylabel("Mean Value")
plt.title("Comparison of Mean Values across Periods")

# Formatting y-axis ticks as percentages
plt.gca().set_yticklabels(["{:.2f}%".format(x * 100) for x in plt.gca().get_yticks()])

# Adding the numeric values on top of each bar
for bar, mean_val in zip(bars, mean_values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        "{:.2f}%".format(mean_val * 100),
        ha="center",
        va="bottom",
    )

plt.show()
# %%
# Extracting the 'std' values from each Series
std_values = [
    stats_period_1.loc["std"],
    stats_period_2.loc["std"],
    stats_period_3.loc["std"],
    stats_period_4.loc["std"],
]

# Creating a bar plot for standard deviations
plt.figure(figsize=(8, 6))
bars = plt.bar(periods, std_values, color="lightgreen")
plt.xlabel("Periods")
plt.ylabel("Standard Deviation Value")
plt.title("Comparison of Standard Deviations across Periods")

# Formatting y-axis ticks as percentages
plt.gca().set_yticklabels(["{:.2f}%".format(x * 100) for x in plt.gca().get_yticks()])

# Adding the numeric values on top of each bar
for bar, std_val in zip(bars, std_values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        "{:.3f}%".format(std_val * 100),
        ha="center",
        va="bottom",
    )

plt.show()

# %%
# Excess returns:
# Mean:
# Extracting the 'mean' values from each Series
means = [
    stats_er_period_1.loc["mean"],
    stats_er_period_2.loc["mean"],
    stats_er_period_3.loc["mean"],
    stats_er_period_4.loc["mean"],
]

# Creating labels for the x-axis (periods)
periods = ["2001-2005", "2006-2010", "2011-2015", "2016-2022"]

# Creating a bar plot
plt.figure(figsize=(8, 6))
bars = plt.bar(periods, means, color="skyblue")
plt.xlabel("Periods")
plt.ylabel("Mean Value")
plt.title("Comparison of Mean Values of Excess Return across Periods")

# Formatting y-axis ticks as percentages
plt.gca().set_yticklabels(["{:.3f}%".format(x * 100) for x in plt.gca().get_yticks()])

# Adding the numeric values on top of each bar
for bar, mean in zip(bars, means):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        "{:.3f}%".format(mean * 100),
        ha="center",
        va="bottom",
    )

plt.show()

# %%
# Standard deviation:
# Extracting the 'std' values from each Series
std_values = [
    stats_er_period_1.loc["std"],
    stats_er_period_2.loc["std"],
    stats_er_period_3.loc["std"],
    stats_er_period_4.loc["std"],
]

# Creating a bar plot for standard deviations
plt.figure(figsize=(8, 6))
bars = plt.bar(periods, std_values, color="lightgreen")
plt.xlabel("Periods")
plt.ylabel("Standard Deviation Value")
plt.title("Comparison of Standard Deviations of Excess Return across Periods")

# Formatting y-axis ticks as percentages
plt.gca().set_yticklabels(["{:.2f}%".format(x * 100) for x in plt.gca().get_yticks()])

# Adding the numeric values on top of each bar
for bar, std in zip(bars, std_values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        "{:.3f}%".format(std * 100),
        ha="center",
        va="bottom",
    )

plt.show()

# %%
# Sharpe ratio:
# Extracting the 'sharpe_ratio' values from each Series
sharpe_ratios = [
    stats_er_period_1.get("sharpe_ratio", float("nan")),
    stats_er_period_2.get("sharpe_ratio", float("nan")),
    stats_er_period_3.get("sharpe_ratio", float("nan")),
    stats_er_period_4.get("sharpe_ratio", float("nan")),
]

# Creating a bar plot for Sharpe ratios
plt.figure(figsize=(8, 6))
bars = plt.bar(periods, sharpe_ratios, color="salmon")
plt.xlabel("Periods")
plt.ylabel("Sharpe Ratio Value")
plt.title("Comparison of Sharpe Ratios across Periods")

# Adding the numeric values on top of each bar
for bar, sharpe_ratio in zip(bars, sharpe_ratios):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        "{:.3f}%".format(sharpe_ratio),
        ha="center",
        va="bottom",
    )

plt.show()

# %%
