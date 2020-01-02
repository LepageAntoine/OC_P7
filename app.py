# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table
import plotly.express as px
import plotly.graph_objects as go

import pickle

import pandas as pd

data_dash = pd.read_csv('data/data_dash.csv')
credit_balance = pd.read_csv('data/credit_card_balance.csv')

# Réduction du nombre de clients dans la base pour accélerer le dashboard en en gardant que les clients présents dans credit_card_balance
data_dash = data_dash.loc[data_dash['SK_ID_CURR'].isin(credit_balance['SK_ID_CURR'].unique())]
data_dash.index = data_dash.SK_ID_CURR.astype(str)




info_cols = ['AGE', 'CODE_GENDER', 'OCCUPATION_TYPE', 'DAYS_EMPLOYED_YEARS', 'NAME_FAMILY_STATUS', 'NB_OF_CHILDREN', 'HOMEOWNER']
situation_cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT_WANTED', 'REMAINING_DEBT', 'REMAINING_DEBT_DURATION(YEARS)', 'NOTATION_EXTERIEUR']
infos_clients = data_dash[info_cols].head(1).T
situtation_fin = data_dash[situation_cols].head(1).T
infos_clients.columns = ['info client']
situtation_fin.columns = ['info client']
infos_clients = infos_clients.reset_index()
situtation_fin = situtation_fin.reset_index()

# Filtre sur les 2 dernieres années
#credit_balance=credit_balance.loc[credit_balance['MONTHS_BALANCE'] > -24]


# liste des numéros clients
sk_id = data_dash['SK_ID_CURR'].unique()

# plot histogrammes généraux
fig = px.histogram(data_dash, x="AMT_INCOME_TOTAL")
fig_2 = px.histogram(data_dash, x="DAYS_EMPLOYED_YEARS")

# load the scoring-model from disk
loaded_model = pickle.load(open('C:/Users/Yop1001/Documents/Cours OC/Parcours Data Scientist/Projet 7 - Implémentez un modèle de scoring/finalized_model.sav', 'rb'))

# load the vectors for modelisation
final_df_modelisation = pd.read_csv('C:/Users/Yop1001/Documents/Cours OC/Parcours Data Scientist/Projet 7 - Implémentez un modèle de scoring/04 - Dashboards/data/final_df_modelisation.csv', index_col=0)


#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app = dash.Dash()

app.layout = html.Div([
	
	# Bloc niveau 1 - bandeau titre
	html.Div([
		# Affichage des infos clients selon sk_id
		html.Div([
			dcc.Dropdown(
		        id='sk_id',
		        options=[{'label': i, 'value': i} for i in sk_id],
		        value = '100028'
		    )
		],className="one-third column"),

		html.Div([
			html.H1('DEMANDE DE PRÊT IMMOBILIER'),
		],className="one-third column"),

		html.Div([
            html.Img(
                src=app.get_asset_url("fake_bank_logo.png"),
                id="fake_bank_logo",
                style={
                	"float": "right",
                    "height": "50px",
                    "width": "auto",
                    "margin-bottom": "20px",
                	"align-items": "start"
                },
            )
        ],className="one-third column"),


    ],className="row"),

	# Bloc niveau 1 - corps
	html.Div([


		# Bloc niveau 2 - colonne gauche
		html.Div([

			html.Div([
				html.H2('Informations clients'),
			]),	

			html.Div([
				dash_table.DataTable(
		    		id='table_infos_clients',
		    		columns=[{"name": i, "id": i} for i in infos_clients.columns],
		    		data=infos_clients.to_dict('records'),
		    		style_cell_conditional=[
				        {'if': {'column_id': 'index'},
				         'width': '60%'},
				        {'if': {'column_id': 'info client'},
				         'width': '40%'},
				    ],
		    		style_cell={'textAlign': 'left'},
		    		style_as_list_view=True,
		    		style_header={
				        'backgroundColor': 'white',
				        'fontWeight': 'bold'
				    },
		    	)
			]),
			
			html.Div([
				html.H2('Situation financière'),
			]),	

			html.Div([
				dash_table.DataTable(
		    		id='table_situation_financiere',
		    		columns=[{"name": i, "id": i} for i in situtation_fin.columns],
		    		data=situtation_fin.to_dict('records'),
		    		style_cell_conditional=[
				        {'if': {'column_id': 'index'},
				         'width': '60%'},
				        {'if': {'column_id': 'info client'},
				         'width': '40%'},
				    ],
		    		style_cell={'textAlign': 'left'},
		    		style_as_list_view=True,
		    		style_header={
				        'backgroundColor': 'white',
				        'fontWeight': 'bold'
				    },
		    	)
			]),

			html.Div([
				html.Div([
					html.H2('Balance mensuelle'),
				]),	

				html.Div(id='balance_mensuelle_graph')
			]),		
		], className="pretty_container one-third column"),



		# Bloc niveau 2 - colonne milieu
		html.Div([

			html.Div([
				html.H2('Calcul de la notation'),
			]),	

			html.Div([
				html.H3('Montant souhaité'),

				dcc.Input(
					id='input_montant_souhaite',
				    placeholder='Enter a value...',
				    type='number',
				    value='1'
				)
			], className="row"),

			html.Div([
				html.H3('Revenu'),

				dcc.Input(
				    id='input_revenu',
				    placeholder='Enter a value...',
				    type='number',
				    value='1'
				) 
			]),	
			
			html.Div([
				html.H2('Notation'),
				html.H3(id='score')
			])
		], className="pretty_container one-third column"),	

		# Bloc niveau 2 - colonne droite
		html.Div([

			html.Div([
				html.H2('Comparaison'),
			]),

			html.Div([
				html.H3(id='output_RangeSlider_age'),

				dcc.RangeSlider(
				    id='RangeSlider_age',
				    min=data_dash['AGE'].min(),
				    max=data_dash['AGE'].max(),
				    step=1,
				    marks={
				        20: '20 ans',
				        30: '30 ans',
				        40: '40 ans',
				        50: '50 ans',
				        60: '60 ans',
				        70: '70 ans'
				    },
				    value=[20, 70]
				)
			]),

			html.Div([
				dcc.Graph(id = 'hist_revenu', figure = fig),
			]),

			html.Div([
				dcc.Graph(id = 'hist_anciennete', figure = fig_2),
			])

		], className="pretty_container one-third column")
	],className="row")
])


# Callbacks

# MAJ df informations client
@app.callback(Output('table_infos_clients', 'data'),
              [Input('sk_id', 'value')])
def filter_table(sk_id_filter):
	filtered_df = data_dash.loc[str(sk_id_filter), info_cols].to_frame()
	filtered_df.columns = ['info client']
	filtered_df = filtered_df.reset_index()
	return filtered_df.to_dict('rows')

# MAJ df Situation client
@app.callback(Output('table_situation_financiere', 'data'),
              [Input('sk_id', 'value')])
def filter_table(sk_id_filter):
	filtered_df = data_dash.loc[str(sk_id_filter), situation_cols].to_frame()
	filtered_df.columns = ['info client']
	filtered_df = filtered_df.reset_index()
	return filtered_df.to_dict('rows')

# MAJ balance mensuelle
@app.callback(Output('balance_mensuelle_graph', 'children'),
              [Input('sk_id', 'value')])
def update_figure(selected_sk_id):
    filtered_df = credit_balance[credit_balance['SK_ID_CURR'] == selected_sk_id]

    return dcc.Graph(
		id='balance_mensuelle',
		figure={
			'data':[{'x': filtered_df['MONTHS_BALANCE'], 'y' : filtered_df['AMT_BALANCE'], 'type': 'bar', "marker": {"color": "#a81e1e"}}],
			'layout' : {
    			'title' : 'Balance mensuelle',
    			'xaxis':{
                    'title':'mois'
                },
                'yaxis':{
                     'title':'(monnaie)'
                }
    		}
		}
	)


# MAJ du score
@app.callback(Output('score', 'children'),
              [Input('sk_id', 'value'), Input('input_montant_souhaite', 'value'), Input('input_revenu', 'value')])
def update_score(selected_sk_id, montant_souhaite, revenu):
	vector = final_df_modelisation.loc[(final_df_modelisation.index == selected_sk_id)]
	vector.loc[:, 'AMT_CREDIT'] = montant_souhaite
	vector.loc[:, 'AMT_INCOME_TOTAL'] = revenu

	score = loaded_model.predict_proba(vector.values)
	score = score[0][1]
	score = score.round(2)

	return score

# Affichage de la tranche d'age du RangeSlider
@app.callback(Output('output_RangeSlider_age', 'children'),
              [Input('RangeSlider_age', 'value')])
def show_rng_slider_max_min(numbers):
    if numbers is None:
        raise PreventUpdate
    return 'Age : de ' + ' à: '.join([str(numbers[0]), str(numbers[-1])])

# MAJ Histogramme Revenu
@app.callback(Output('hist_revenu', 'figure'),
              [Input('RangeSlider_age', 'value'), Input('sk_id', 'value')])
def update_figure(numbers, sk_id):
    filtered_df_3 = data_dash.loc[(data_dash['AGE'] > numbers[0]) & (data_dash['AGE'] < numbers[-1])]
    fig = px.histogram(filtered_df_3, x="AMT_INCOME_TOTAL", title='Histogramme des revenus', labels={'AMT_INCOME_TOTAL':'Revenu'})
    fig.add_shape(
    	go.layout.Shape(
    		type='line', 
    		xref='x', 
    		yref='y',
            x0=data_dash.loc[data_dash['SK_ID_CURR'] == sk_id, 'AMT_INCOME_TOTAL'].values[0], 
            y0=0, 
            x1=data_dash.loc[data_dash['SK_ID_CURR'] == sk_id, 'AMT_INCOME_TOTAL'].values[0], 
            y1=400, 
            line={'dash': 'dash'}),
    )

    return fig




# MAJ Histogramme Anciennete
@app.callback(Output('hist_anciennete', 'figure'),
              [Input('RangeSlider_age', 'value')])
def update_figure(numbers):
    filtered_df_2 = data_dash.loc[(data_dash['AGE'] > numbers[0]) & (data_dash['AGE'] < numbers[-1])]
    fig = px.histogram(filtered_df_2, x="DAYS_EMPLOYED_YEARS", title='Histogramme de l\'ancienneté', labels={'DAYS_EMPLOYED_YEARS':'Ancienneté (année)'})
    #fig.add_shape(
    #go.layout.Shape(type='line', xref='x', yref='paper',
    #                x0=data_dash.loc[data_dash['SK_ID_CURR'] == , ], y0=0, x1=0, y1=20000, line={'dash': 'dash'}),
    #)

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)


