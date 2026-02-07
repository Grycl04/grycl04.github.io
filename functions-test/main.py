import functions_framework

@functions_framework.http
def health(request):
    return ("{\"status\": \"healthy\"}", 200, {"Content-Type": "application/json"})
