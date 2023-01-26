from pandas import read_csv
from matplotlib.pyplot import figure, xticks, show, savefig
import matplotlib.pyplot as plt
from ts_functions import plot_series, HEIGHT

data = read_csv('../glucose.csv', index_col='Date', sep=',', decimal='.', parse_dates=True, dayfirst=True)
print("Nr. Records = ", data.shape[0])
print("First timestamp", data.index[0])
print("Last timestamp", data.index[-1])
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(data, x_label='timestamp', y_label='glucose level', title='Glucose')
plt.gca().legend(('glucose'))
xticks(rotation = 45)
show()
savefig("imagesD1Granularity/test1.png")


day_df = data.copy().groupby(data.index.date).mean()
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(day_df, title='Daily', x_label='timestamp', y_label='glucose level')
plt.gca().legend(('glucose'))
xticks(rotation = 45)
show()
savefig("imagesD1Granularity/daily.png")


index = data.index.to_period('W')
week_df = data.copy().groupby(index).mean()
week_df['timestamp'] = index.drop_duplicates().to_timestamp()
week_df.set_index('timestamp', drop=True, inplace=True)
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(week_df, title='Weekly', x_label='timestamp', y_label='glucose level')
plt.gca().legend(('glucose'))
xticks(rotation = 45)
show()
savefig("imagesD1Granularity/weekly.png")

index = data.index.to_period('H')
month_df = data.copy().groupby(index).mean()
month_df['timestamp'] = index.drop_duplicates().to_timestamp()
month_df.set_index('timestamp', drop=True, inplace=True)
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(month_df, title='Hourly', x_label='timestamp', y_label='glucose level')
plt.gca().legend(('glucose'))
show()
savefig("imagesD1Granularity/hourly.png")