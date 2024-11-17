
from re import X
from django.shortcuts import render
import pandas as pd  # type: ignore
from django.conf import settings
from rest_framework.views import APIView  # type: ignore
from rest_framework.response import Response  # type: ignore
from rest_framework import status  # type: ignore
import pickle

# Carregar o modelo treinado
modelo_path = "./clinica/models/modelo_teste.pkl"
colunas_path = "./clinica/models/colunas.pkl"
try:
    # Carrega o modelo
    with open(modelo_path, 'rb') as f:
        modelo = pickle.load(f)
        print("Modelo carregado com sucesso!")
    with open(colunas_path, 'rb') as f:
        colunas_esperadas = pickle.load(f)
        print("Colunas carregadas com sucesso!")    
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    exit()

class CirrosePredictionView(APIView):
    def post(self, request):
        # Recebe os dados do paciente
        data = request.data
        
        # Converte dados do paciente para DataFrame
        entrada_usuario = pd.DataFrame([data])
        
        try:
            entrada_usuario['Sex'] = entrada_usuario['Sex'].map({'M': 1, 'F': 0})  # Mapear Sexo para numérico
            entrada_usuario = entrada_usuario.reindex(columns=colunas_esperadas, fill_value=0)
            print("Dados processados para entrada no modelo:")
            print(entrada_usuario)

            previsao = modelo.predict(entrada_usuario)

            return Response(
            {f"Com base nos dados fornecidos, seu estágio de cirrose previsto é: {int(previsao[0])}"},
            status=status.HTTP_200_OK
            )
        except Exception as e:
            return Response({"Erro durante o processamento:": (e)}, status=status.HTTP_400_BAD_REQUEST)

