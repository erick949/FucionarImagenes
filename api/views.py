from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import HttpResponse
from .services.stitching import unir_imagenes_bytes


@api_view(['POST'])
def unir_imagenes_view(request):
    try:
        img1 = request.FILES.get('img1')
        img2 = request.FILES.get('img2')

        if not img1 or not img2:
            return Response({"error": "Se requieren dos imágenes"}, status=400)

        resultado = unir_imagenes_bytes(img1.read(), img2.read())

        return HttpResponse(resultado, content_type="image/jpeg")

    except Exception as e:
        return Response({"error": str(e)}, status=500)