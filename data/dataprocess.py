import pandas as pd

temp_list = ['0C', '10C', '25C', '30C', '40C', '50C', 'RT']
mod_list = ['DST', 'FUDS', 'US06']

for temp in temp_list:
    for mod in mod_list:
        path = f'./{temp}/{mod}_{temp}.csv'
        data = pd.read_csv(path)
        date = pd.date_range('2022-01-01', periods=len(data), freq='1s')
        date = date.strftime('%X')
        col = list(data.columns)
        col.remove('Profile')
        data = data[col]
        data.insert(loc=0, column='date', value=0)
        data['date'] = date

        # print(data)
        data.to_csv(f'./{temp}/{mod}_{temp}_data.csv', index=0)
