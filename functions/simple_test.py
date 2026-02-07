from firebase_functions import https_fn

@https_fn.on_request()
def hello_world(request):
    return "Hello World from Firebase!"
