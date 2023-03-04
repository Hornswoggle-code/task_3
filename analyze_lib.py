from dask import dataframe as dd


def read_files():
    no_promotion = dd.read_csv('final_data/transactions_no_promotion.csv/*')
    no_promotion['TransactionDate'] = dd.to_datetime(no_promotion['TransactionDate'])
    with_promotion = dd.read_csv('final_data/transactions_with_promotion.csv/*')
    with_promotion['TransactionDate'] = dd.to_datetime(with_promotion['TransactionDate'])
    return dd.concat([no_promotion, with_promotion[no_promotion.columns]])


def split_dates(dataframe):
    holidays = dd.read_csv('holidays/all_holidays.csv')
    holidays['Date'] = dd.to_datetime(holidays['Date'], dayfirst=True)
    holidays = holidays['Date'].values.compute()
    split_dates_holidays(dataframe, holidays)


def split_dates_holidays(dataframe, holidays):
    dataframe['Week'] = dataframe['TransactionDate'].dt.to_period('W')
    dataframe['Month'] = dataframe['TransactionDate'].dt.to_period('M')
    dataframe['Year'] = dataframe['TransactionDate'].dt.to_period('Y')
    dataframe['Quarter'] = dataframe['TransactionDate'].dt.to_period('Q')
    dataframe['IsHoliday'] = dataframe['TransactionDate'].isin(holidays)


def sum_values_groupby_to_csv(dataframe, group, column, name):
    sum_values_groupby_sort(dataframe, group, column).to_csv(name, index=False)


def sum_values_groupby_sort(dataframe, group, column):
    return sum_values_groupby_c(dataframe, group, column).reset_index().sort_values(column, ascending=False)


def sum_values_groupby_c(dataframe, group, column):
    return dataframe.groupby(group)[column].sum().compute()


def sum_values_groupby(dataframe, group, column):
    return dataframe.groupby(group)[column].sum()


def mean_values_groupby(dataframe, group, column):
    return dataframe.groupby(group)[column].mean()
