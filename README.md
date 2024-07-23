# UniSwipeBackend
Python REST API Backend for UniSwipe App

## Getting Started

PREREQUISITES: 
node 
npm package manager 
Python 3.7+

Clone the repository: https://github.com/kalxed/uniswipe-backend.git

Navigate to the root directory using cd, then run: 
```bash
pip install -r requirements.txt
```
This will install the required pip packages for the project. Then run:
```bash 
python api.py
```
Make note of the URL that is used for the flask server locally, this will be used to connect to the frontend.
## Connecting to Frontend 

The frontend is available for download here: https://github.com/kalxed/uniswipe. 

There, navigate to the root directory and run 
``` 
npm install . 
```
Before running, change the `.env.local` variable to match the URL that was printed to the console when running the python API.

Then, to start the server, run: 
```
npm run dev 
```

Now you are able to run the website with the given port, and you should be able to retrieve schools and swipe through them on the site using the model. 