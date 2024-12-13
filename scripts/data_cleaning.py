import pandas as pd
def clean(data):
    check = data.isnull().sum()
    print(check)
    print(f"There is {data.duplicated().sum()} duplicate")
    data['Date'] = pd.to_datetime(data['Date'])
   
import sys
print(sys.path)