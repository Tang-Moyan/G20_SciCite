import pandas as pd

csv_file_path = './trainOriginal.csv'  # Make sure to update this path to where your CSV file is located

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

def find_length(string):
    # print(int(len(string) / 100) * 100)
    fixed_len = 170
    if len(string) < fixed_len:
        return 'Short'
    if len(string) < 250:
        return 'Medium'
    return 'Long'
    # return int(len(string) / 100) * 100

df['length'] = df['string'].apply(find_length)

# print(df['length'].quantile(0.33))
# print(df['length'].quantile(0.66))

label_counts = df.groupby('label')['length'].value_counts()
print(label_counts)

df_short = df[df['length'] == 'Short']
df_medium = df[df['length'] == 'Medium']
df_long = df[df['length'] == 'Long']

df_short.to_csv('./data_files/short_len_train.csv', index=False)
df_medium.to_csv('./data_files/med_len_train.csv', index=False)
df_long.to_csv('./data_files/long_len_train.csv', index=False)
df.to_csv('./data_files/overall_len_train.csv', index=False)

