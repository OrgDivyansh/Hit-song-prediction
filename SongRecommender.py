import pandas as pd
import numpy as np
pd.set_option('display.width', 500)      
pd.set_option('display.max_colwidth', 20)  
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


# 1. Load the dataset
df = pd.read_csv('spotify-tracks-dataset-detailed.csv')

features = ['danceability', 'energy', 'valence', 'tempo', 'loudness']

columns_to_keep = features + ['track_name', 'artists']
df_rec = df[columns_to_keep].dropna()
df_rec = df[columns_to_keep].dropna().drop_duplicates(subset=['track_name', 'artists'])


print(f"Dataset ready! We have {len(df_rec)} songs to recommend from.")

# 2. Initialiseing the scaler
scaler = StandardScaler()

scaled_features = scaler.fit_transform(df_rec[features])
print("Features scaled successfully!")

# 3. Building the Cosine similarity model
n_neighbours = 6

knn_model = NearestNeighbors(n_neighbors=n_neighbours, metric= 'cosine', algorithm= 'brute')
knn_model.fit(scaled_features)

print("Recommender model is trained and ready!")



# 4. Now creating the recommendation function
def recommend_songs(song_name, df, model, scaled_data, num_recs=5):

    matches = df[df['track_name'].str.lower() == song_name.lower()]

    if matches.empty:
        return f" Song '{song_name}' not found! check the spelling bub."

    song_idx = matches.index[0]

    distances, indices = model.kneighbors([scaled_data[song_idx]], n_neighbours+num_recs + 1)
    rec_indices = indices[0][1:]

    recommendations = df.iloc[rec_indices][['track_name','artists']]

    
    print(f"\n🎵 Top {num_recs} recommendation based on '{song_name}':\n")
    return recommendations

#5. Test
def run_recommender_app():
    print("🎧 Welcome to the Spotify audio feature recommender! 🎧")
    print("Type the name of the song you like ,I'll find the similar tracks")
    print("(Type 'quit' or 'exit'to stop)\n")

    while True:
        user_song = input("Enter a song name:").strip()

        if user_song.lower() in ['quit', 'exit', 'q']:
           print("\nThanks for listening! Goodbye.")
           break

        if not user_song:
            continue

        #allows user to search for partial names

        if user_song.lower() in ['search', 'find', 'help']:
            keyword = input("Enter a keyword to search for in song titles: ").strip()
            
            # Search the dataframe for partial matches
            matches = df_rec[df_rec['track_name'].str.contains(keyword, case=False)].head(10)
            
            if not matches.empty:
                print(f"\n🔍 Found these songs containing '{keyword}':\n")
                # Print them using the clean list format
                for index, row in matches.iterrows():
                    print(f"👉 {row['track_name']}  |  {row['artists']}")
            else:
                print("❌ No songs found with that keyword.")
                
            # 'continue' skips the rest of the loop and goes back to the top to ask for input again
            continue 
        #============================================   
        result = recommend_songs(user_song, df_rec, knn_model, scaled_features, num_recs=5)

        if isinstance(result, pd.DataFrame):
            print(result.to_string(index= False))

        else:
            print(result)

        print("-" * 50)

# A basic web interface code for song recommendation

import gradio as gr

# 1. Create a wrapper function for Gradio to use
def get_recommendations_ui(song_name):
    result = recommend_songs(song_name, df_rec, knn_model, scaled_features, num_recs=5)
    
    # Format the output nicely as text
    if isinstance(result, pd.DataFrame):
        return "\n".join([f"👉 {row['track_name']} | {row['artists']}" for _, row in result.iterrows()])
    return result

# 2. Build the Web UI in just a few lines
gr.Interface(
    fn=get_recommendations_ui, 
    inputs=gr.Textbox(label="Enter a song name", placeholder="e.g. Blinding Lights"), 
    outputs=gr.Textbox(label="Recommended Songs", lines=10, interactive=False),
    title="🎧 Spotify Audio Recommender",
    description="Find songs with similar vibes based on audio features!"
).launch('share = True')