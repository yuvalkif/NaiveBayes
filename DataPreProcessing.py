import pandas as pd

def fillDatasetNANumerical(dataset):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    meansDf = dataset.select_dtypes(include=numerics)#only numeric cols
    numericNames = list(meansDf)#store numeric names
    meansDf = meansDf.mean(skipna=True)

    for colName in numericNames:
        dataset[colName].fillna(meansDf[colName],inplace=True)

    return dataset


def fillDatasetNACategorical(dataset):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    categoricalNames = dataset.select_dtypes(exclude=numerics)

    for colName in categoricalNames:
        dataset[colName].fillna(dataset[colName].mode()[0],inplace=True)

    return dataset


def getNormalizedValue(value , min , max):

    newmin = 0
    newmax = 100

    value = ((value-min)/(max-min))*(newmax-newmin)+newmin
    return value

def getEqualWidthBins(numofbins):

    width = 100/numofbins
    bins = [0]
    start = 0

    while(numofbins > 0):

        bins.append(start+width)
        start = start + width
        numofbins = numofbins - 1

    return bins

def discretizeDataset(dataset , numofbins):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numericalNames = dataset.select_dtypes(include=numerics)

    width = 100/numofbins
    bins = getEqualWidthBins(numofbins)
    labels = []
    num = 1

    while(num <= numofbins):
        labels.append(str(num))
        num = num+1

    pd.options.mode.chained_assignment = None

    for colName in list(numericalNames.columns):
        max = dataset[colName].max()
        min = dataset[colName].min()

        for index,value in pd.DataFrame(dataset[colName]).iterrows():
            dataset[colName][index] = getNormalizedValue(value[0],min,max)
        #do the discretization

        dataset[colName] = pd.cut(dataset[colName],bins=numofbins,labels=labels)


    return dataset


