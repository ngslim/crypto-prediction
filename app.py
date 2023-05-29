import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from sklearn.preprocessing import MinMaxScaler

app = dash.Dash()
server = app.server

def LSTM_Prediction(data_csv):
    df=pd.read_csv(data_csv)

    df["Date"]=pd.to_datetime(df.Date,format="%Y-%m-%d")
    df.index=df['Date']

    data=df.sort_index(ascending=True,axis=0)
    new_dataset=pd.DataFrame(index=range(0,len(df)),columns=['Date','Close'])
    for i in range(0,len(data)):
        new_dataset["Date"][i]=data['Date'][i]
        new_dataset["Close"][i]=data["Close"][i]
    new_dataset['Date'] = pd.to_numeric(pd.to_datetime(new_dataset['Date']))

    scaler=MinMaxScaler(feature_range=(0,1))
    final_dataset=new_dataset.values
    train_data=final_dataset[0:987,:]
    valid_data=final_dataset[987:,:]
    new_dataset.index=new_dataset.Date
    new_dataset.drop("Date",axis=1,inplace=True)
    scaler=MinMaxScaler(feature_range=(0,1))
    scaled_data=scaler.fit_transform(final_dataset)
    x_train_data,y_train_data=[],[]
    for i in range(60,len(train_data)):
        x_train_data.append(scaled_data[i-60:i,0])
        y_train_data.append(scaled_data[i,0])
        
    x_train_data,y_train_data=np.array(x_train_data),np.array(y_train_data)
    x_train_data=np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1))

    lstm_model=Sequential()
    lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train_data.shape[1],1)))
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(1))
    inputs_data=new_dataset[len(new_dataset)-len(valid_data)-60:].values
    inputs_data=inputs_data.reshape(-1,1)
    inputs_data=scaler.fit_transform(inputs_data)
    lstm_model.compile(loss='mean_squared_error',optimizer='adam')
    lstm_model.fit(x_train_data,y_train_data,epochs=1,batch_size=1,verbose=2)

    lstm_model.save(data_csv[0:-4] + ".h5")

    X_test=[]
    for i in range(60,inputs_data.shape[0]):
        X_test.append(inputs_data[i-60:i,0])
    X_test=np.array(X_test)
    X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
    predicted_closing_price=lstm_model.predict(X_test)
    predicted_closing_price=scaler.inverse_transform(predicted_closing_price)

    train_data=new_dataset[:987]
    valid_data=new_dataset[987:]
    train_data.index = pd.to_datetime(train_data.index) 
    valid_data.index = pd.to_datetime(valid_data.index) 
    valid_data['Predictions']=predicted_closing_price
    
    return train_data, valid_data

BTC_train, BTC_valid = LSTM_Prediction("BTC-USD.csv")
ETH_train, ETH_valid = LSTM_Prediction("ETH-USD.csv")
ADA_train, ADA_valid = LSTM_Prediction("ADA-USD.csv")

app.layout = html.Div([
   
    html.H1("Cryptocurrencies Price Analysis Dashboard", style={"textAlign": "center"}),
   
    dcc.Tabs(id="tabs", children=[
       
        dcc.Tab(label='BTC-USD',children=[
            html.Div([
                html.H2("Actual Closing Price",style={"textAlign": "center"}),
                dcc.Graph(
                    id="Actual Data",
                    figure={
                        "data":[
                            go.Scatter(
                                x=BTC_valid.index,
                                y=BTC_valid["Close"],
                                mode='markers'
                            )
                        ],
                        "layout":go.Layout(
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Rate'}
                        )
                    }
                ),
                html.H2("LSTM Predicted Closing Price",style={"textAlign": "center"}),
                dcc.Graph(
                    id="Predicted Data",
                    figure={
                        "data":[
                            go.Scatter(
                                x=BTC_valid.index,
                                y=BTC_valid["Predictions"],
                                mode='markers'
                            )
                        ],
                        "layout":go.Layout(
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Rate'}
                        )
                    }
                )                
            ])                
        ]),

        dcc.Tab(label='ETH-USD',children=[
            html.Div([
                html.H2("Actual Closing Price",style={"textAlign": "center"}),
                dcc.Graph(
                    id="Actual Data",
                    figure={
                        "data":[
                            go.Scatter(
                                x=ETH_valid.index,
                                y=ETH_valid["Close"],
                                mode='markers'
                            )
                        ],
                        "layout":go.Layout(
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Rate'}
                        )
                    }
                ),
                html.H2("LSTM Predicted Closing Price",style={"textAlign": "center"}),
                dcc.Graph(
                    id="Predicted Data",
                    figure={
                        "data":[
                            go.Scatter(
                                x=ETH_valid.index,
                                y=ETH_valid["Predictions"],
                                mode='markers'
                            )
                        ],
                        "layout":go.Layout(
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Rate'}
                        )
                    }
                )                
            ])                
        ]),

        dcc.Tab(label='ADA-USD',children=[
            html.Div([
                html.H2("Actual Closing Price",style={"textAlign": "center"}),
                dcc.Graph(
                    id="Actual Data",
                    figure={
                        "data":[
                            go.Scatter(
                                x=ADA_valid.index,
                                y=ADA_valid["Close"],
                                mode='markers'
                            )
                        ],
                        "layout":go.Layout(
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Rate'}
                        )
                    }
                ),
                html.H2("LSTM Predicted Closing Price",style={"textAlign": "center"}),
                dcc.Graph(
                    id="Predicted Data",
                    figure={
                        "data":[
                            go.Scatter(
                                x=ADA_valid.index,
                                y=ADA_valid["Predictions"],
                                mode='markers'
                            )
                        ],
                        "layout":go.Layout(
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Rate'}
                        )
                    }
                )                
            ])                
        ]),
    ])
])

if __name__=='__main__':
    app.run_server(debug=True)