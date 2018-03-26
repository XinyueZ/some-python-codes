from media import Movie
from fresh_tomatoes import open_movies_page as show_movies

def generate_page():
    saw_iv = Movie( "SAW IV",
                    "https://upload.wikimedia.org/wikipedia/en/3/35/Saw4final.jpg",
                    "https://www.youtube.com/watch?v=PYQ75b89PPw")

    it = Movie( "It",
                "https://upload.wikimedia.org/wikipedia/en/5/5a/It_%282017%29_poster.jpg",
                "https://www.youtube.com/watch?v=hAUTdjf9rko")

    return [saw_iv, it]

show_movies(generate_page())
