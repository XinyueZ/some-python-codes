class Media():
    def __init__(self, media_title):
        self.title = media_title

    def show_media_info(self):
        print("Title: {}".format(self.title))

class Movie(Media):
    def __init__(self, movie_title, movie_poster_image_url, movie_trailer_youtube_url):
        super().__init__(movie_title)
        self.poster_image_url = movie_poster_image_url
        self.trailer_youtube_url = movie_trailer_youtube_url

    def show_media_info(self):
        super().show_media_info()
        print("poster_image_url: {}".format(self.poster_image_url))
        print("trailer_youtube_url: {}".format(self.trailer_youtube_url))
