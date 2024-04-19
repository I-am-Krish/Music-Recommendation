from flask import Flask, request, render_template
from src.pipeline.prediction_pipeline import MusicRecommendationModel

application = Flask(__name__)
app = application

recommendation_model = MusicRecommendationModel()
recommendation_model.load_pretrained_model()

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/recommend_song', methods=['POST', 'GET'])
def recommend_song():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        song_name = request.form.get('song_name')
        year = request.form.get('year')
        artists = request.form.get('artists')
        key = request.form.get('key')
        ID = request.form.get('ID')

        artists_list = artists.replace('[', '').replace(']', '').replace("'", "").split(', ')

        input_data = {
            'name': [song_name],
            'year': [year],
            'artists': [artists_list],
            'key': [int(key)],
            'ID': [int(ID)]
        }

        recommended_songs = recommendation_model.recommend_songs(song_name)

        return render_template('results.html', song_name=song_name, final_result=recommended_songs)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

#http://127.0.0.1:5000/ in browser
