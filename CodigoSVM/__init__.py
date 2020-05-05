import pandas as pd
import numpy as np
import pyodbc
import json
import os
import azure.functions as func
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

def main(req: func.HttpRequest) -> func.HttpResponse:
    driverAzure = os.environ["DriverAzure"]
    serverAzure = os.environ["ServerBdAzure"]
    databaseAzure = os.environ["DataBaseAzure"]
    usernameAzure = os.environ["UserNameBdAzure"]
    passwordAzure = os.environ["PassWordBdAzure"]
    SQL_datos = os.environ["SQL_datos"]
    conStringAzure = "DRIVER={{{}}};SERVER={};DATABASE={};UID={};PWD={}".format(driverAzure,serverAzure,databaseAzure,usernameAzure,passwordAzure) #INGRESA LOS DATOS EN ORDEN
    cnxn = pyodbc.connect(conStringAzure)
    req_body = req.get_json()
    variable1 = req_body.get('variable1')
    cursor = cnxn.cursor()
    #LEER Y
    Yquery = "SELECT [Y] FROM logsY;"
    YSQL = pd.read_sql(Yquery, cnxn)
    #LEER X
    Xquery = "SELECT [X1],[X2],[X3],[X4],[X5],[X6],[X7],[X8],[X9],[X10],[X11],[X12] FROM logsX;"
    XSQL = pd.read_sql(Xquery, cnxn)
    #CONVERTIR DATAFRAME DE PANDAS A MATRIZ NUMPY
    Y=YSQL.values
    X=XSQL.values
    #NORMALIZAR
    m,n = np.shape(X)
    minimos = np.min(X,axis=0)
    maximos = np.max(X,axis=0)
    Xn=(X-minimos)/(maximos-minimos)
    #VALIDACION
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xn,Y,test_size=0.3,random_state=42)
    #CREACION Y ENTRENAMIENTO DEL MODELO SVM
    modelo = SVC(kernel='linear',gamma='auto')
    modelo.fit(Xtrain, Ytrain)
    #PREDICCION
    predicciones = modelo.predict(Xtest)
    json_response = json.dumps(classification_report(Ytest, predicciones),indent=2)
    json_response = classification_report(Ytest, predicciones,labels=[-1,0,1])
    if variable1 < 10:
        return func.HttpResponse(json_response)
    else:
        return func.HttpResponse("NUBE Puede que se ingresara un valor mal en el postman pero la funcion se ejecuto bien",status_code=200)