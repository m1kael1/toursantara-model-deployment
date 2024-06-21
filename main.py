from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import pandas as pd
import numpy as np
from model import RecommenderNet

model_path="temp/model/model_recommender"

model = tf.keras.models.load_model(model_path, custom_objects={'RecommenderNet': RecommenderNet})

df = pd.read_csv('data/Tourism Rating.csv')
info_tourism = pd.read_csv('data/Tourism with ID.csv')

print(df)

place_ids = df.Place_Id.unique().tolist()
place_to_place_encoded = {x: i for i, x in enumerate(place_ids)}
place_encoded_to_place = {i: x for i, x in enumerate(place_ids)}

user_ids = df.User_Id.unique().tolist()
user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}

all_tourism = pd.merge(df, info_tourism[["Place_Id", "Place_Name", "Description", "City", "Category", "Rating", "Lat", "Long"]], on='Place_Id', how='left')
preparation = all_tourism.drop_duplicates("Place_Id")

place_df = pd.DataFrame({
    "id": preparation.Place_Id.tolist(),
    "name": preparation.Place_Name.tolist(),
    "category": preparation.Category.tolist(),
    "description": preparation.Description.tolist(),
    "city": preparation.City.tolist(),
    "city_category": (preparation.City + " - " + preparation.Category).tolist(),
    "rating": preparation.Rating.tolist(),
    "lat": preparation.Lat.tolist(), "long": preparation.Long.tolist(), 
})

app = FastAPI()

class RecommendRequest(BaseModel):
    user_id: int

@app.post("/recommend/")
async def recommend_places(request: RecommendRequest):
    user_id = request.user_id

    if user_id not in user_to_user_encoded:
        raise HTTPException(status_code=404, detail="User ID not found")

    user_encoder = user_to_user_encoded.get(user_id)
    place_visited_by_user = df[df.User_Id == user_id]
    print('=====place_visited_by_user=====')
    print(place_visited_by_user)
    place_not_visited = place_df[~place_df['id'].isin(place_visited_by_user['Place_Id'].values)]['id']
    place_not_visited = list(set(place_not_visited).intersection(set(place_to_place_encoded.keys()))) 
    print('=====place_not_visited=====')
    print(place_not_visited)
    place_not_visited = [[place_to_place_encoded.get(x)] for x in place_not_visited]
   
    user_place_array = np.hstack(([[user_encoder]] * len(place_not_visited), place_not_visited))
    

    # Prediksi rating
    ratings = model.predict(user_place_array).flatten()
   
    top_ratings_indices = ratings.argsort()[-10:][::-1]
    print('=====top_ratings_indices=====')
    print(top_ratings_indices)
    recommended_place_ids = [place_encoded_to_place.get(place_not_visited[x][0]) for x in top_ratings_indices]

    # Mengambil detail tempat wisata yang direkomendasikan
    recommended_places = place_df[place_df['id'].isin(recommended_place_ids)]

    result = recommended_places.to_dict(orient='records')

    return {"status": "success", "data": result}

# Menjalankan aplikasi FastAPI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
