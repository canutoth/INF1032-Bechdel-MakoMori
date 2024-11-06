import json
import pandas as pd

with open('all_data.json') as f:
    all_data = json.load(f)

with open('makomori_labeled.json') as f:
    makomori_data = json.load(f)

all_data_df = pd.json_normalize(all_data)
makomori_data_df = pd.DataFrame(makomori_data)

merged_df = pd.merge(all_data_df, makomori_data_df)

final_df = merged_df[['title', 'bechdel', 'imdbid', 'details.imdbRating', 'mako-mori', 'details.BoxOffice']]

final_df = final_df.rename(columns={
    'imdbid': 'imdbID',
    'details.imdbRating': 'rating',
    'mako-mori': 'mako-mori',
    'details.BoxOffice': 'boxOffice'
})

final_json = final_df.to_dict(orient='records')
with open('final.json', 'w') as f:
    json.dump(final_json, f, indent=4)

print("final.json has been created successfully.")