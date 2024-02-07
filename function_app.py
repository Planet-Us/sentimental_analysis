import azure.functions as func
from server import app  # Flask 앱 임포트

def main(req: func.HttpRequest) -> func.HttpResponse:
    # Flask 앱의 WSGI 인터페이스를 Azure Functions 요청으로 래핑합니다.
    return func.WsgiMiddleware(app.wsgi_app).handle(req)
