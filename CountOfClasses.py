import pandas as pd

#Setups data
df = pd.read_csv('C:/Course_work/DSCI_644/Project/QuLog-main/Group_2/Data_In/nine_systems_data.csv')
df = df.drop(df[(df.log_level != "info") & (df.log_level != "warn") & (df.log_level != "error")].index)
#df['static_text'] = df.apply(lambda row: remove_stopwords(row['static_text']), axis=1)

df = df.drop(['project'], axis=1)

print(df.drop(df[df.log_level != "info"].index).count())
print(df.drop(df[df.log_level != "warn"].index).count())
print(df.drop(df[df.log_level != "error"].index).count())