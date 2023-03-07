import os
from heroku3 import from_key

# First, authenticate with Heroku using an API key
HEROKU_API_KEY = os.environ.get('HEROKU_API_KEY')
heroku_conn = from_key(HEROKU_API_KEY)

# Next, create a new Heroku app
app_name = 'docgpt'
heroku_app = heroku_conn.create_app(name=app_name)

# Connect the app to a GitHub repo
github_repo = 'dustinmarshall/docgpt'
heroku_app.create_source_blob('https://github.com/{}/archive/master.tar.gz'.format(github_repo))

# Deploy the app to Heroku
heroku_build = heroku_app.builds.create(source_blob={'url': heroku_app.source_blob.url})
heroku_build.wait_for_build_to_complete()
heroku_app.release(heroku_build.source_version)
