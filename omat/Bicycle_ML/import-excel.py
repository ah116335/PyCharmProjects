import pandas as pd
import numpy as np


def loadexcel(filename, sheetname=0):
    exceldata = None
    df1 = pd.read_excel(filename, sheet_name='Bicycle', usecols="A,J,K")
    exceldata = df1.values
    return exceldata, df1


ExData, ExDFr = loadexcel('bicycle.xlsx')
print(ExDFr.columns)

col1 = ExDFr

# print(ExData[10:12,0])        # print rows 10 and 11 , and column 0 only.  Note - indexing starts from 0
# ExDFr.to_csv('mymymy.csv')    # print a csv of dataframe ExDFr
# np.savetxt('ggg.csv', ExData, delimiter=",", fmt='%s')    #print a csv of numpy array ExData

