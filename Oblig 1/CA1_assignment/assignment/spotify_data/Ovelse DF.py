import pandas as pd
import matplotlib.pyplot as plt

history = pd.read_csv("streaminghistory0.csv")
df_spotify_0 = pd.DataFrame(history)
num_entries = df_spotify_0.shape[0]
new_indexes = [f'Track {x + 1}' for x in range(num_entries)]
df_spotify_0.index = new_indexes
new_headers = {'endTime': 'End Time', 'artistName': 'Artist', 'trackName': 'Track', 'msPlayed': 'Playtime (Ms)'}
df_spotify_0 = df_spotify_0.rename(columns=new_headers)

print(df_spotify_0.dtypes)

df_spotify_0 = df_spotify_0.sort_values(by=['Playtime (Ms)'], ascending=False)

df_spotify_0['End Time'] = pd.to_datetime(df_spotify_0['End Time'])
print(df_spotify_0['End Time'].dt.strftime('%Y-%m-%d %H:%M'))

print(df_spotify_0['End Time'].dt.floor('D'))


print(df_spotify_0['Artist'].unique())



