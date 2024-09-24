from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import json
import plotly

app = Flask(__name__)

# Load and preprocess data
data = pd.read_csv("districtwise-crime-against-women.csv")
crime_columns = [
    'murder_with_rape_or_gang_rape', 'dowry_deaths', 'abetment_to_suicide_of_women',
    'attempt_to_commit_rape', 'assault_on_women_with_intent_to_outrage_her_modesty',
    'insult_to_the_modesty_of_women', 'dowry_prohibition', 'immoral_traffic_prevention_act_total',
    'protection_of_women_from_domestic_violence_act', 'cyber_crimes_or_infor_tech_women_centric_crimes',
    'prot_of_children_frm_sexual_viol_girl_child_victims', 'indecent_representation_of_women_prohibition',
    'total_crime_against_women'
]

imputer = SimpleImputer(strategy='median')
crime_data_imputed = imputer.fit_transform(data[crime_columns])

scaler = StandardScaler()
crime_data_scaled = scaler.fit_transform(crime_data_imputed)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cluster', methods=['POST'])
def cluster():
    n_clusters = int(request.form['n_clusters'])
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    data['cluster'] = kmeans.fit_predict(crime_data_scaled)
    
    # Generate plots
    plots = {}
    
    # Scatter plot
    scatter_fig = px.scatter(
        data, x=crime_data_scaled[:, -1], y=crime_data_scaled[:, 4], color='cluster',
        title='Clusters of Crime Data',
        labels={'x': 'Total Crime Against Women (Scaled)', 'y': 'Assault on Women (Scaled)'}
    )
    plots['scatter'] = json.dumps(scatter_fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Box plot
    box_fig = px.box(
        data, x='cluster', y='total_crime_against_women',
        title='Total Crime Against Women Distribution per Cluster'
    )
    plots['boxplot'] = json.dumps(box_fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Heatmap
    cluster_means = data.groupby('cluster')[crime_columns].mean()
    heatmap_fig = go.Figure(data=go.Heatmap(
        z=cluster_means.values,
        x=cluster_means.columns,
        y=cluster_means.index,
        colorscale='YlOrRd'
    ))
    heatmap_fig.update_layout(title='Average Crime Rates per Cluster')
    plots['heatmap'] = json.dumps(heatmap_fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Bar Chart of Average Crime Rates per Cluster
    bar_avg_fig = px.bar(
        cluster_means.reset_index().melt(id_vars='cluster'), x='variable', y='value', color='cluster',
        title='Average Crime Rates per Cluster', labels={'variable': 'Crime Category', 'value': 'Average Rate'}
    )
    plots['bar_avg'] = json.dumps(bar_avg_fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Violin Plot of a Specific Crime Category
    violin_fig = px.violin(
        data, x='cluster', y='total_crime_against_women', color='cluster',
        title='Distribution of Total Crime Against Women Across Clusters'
    )
    plots['violin'] = json.dumps(violin_fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Histogram of a Specific Crime Category
    hist_fig = px.histogram(
        data, x='total_crime_against_women', color='cluster',
        title='Distribution of Total Crime Against Women'
    )
    plots['histogram'] = json.dumps(hist_fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Scatter Matrix
    scatter_matrix_fig = px.scatter_matrix(
        data, dimensions=crime_columns[:5], color='cluster',
        title='Scatter Matrix of Selected Crime Categories'
    )
    plots['scatter_matrix'] = json.dumps(scatter_matrix_fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Bar Chart of Top 10 Districts by Total Crime
    top_10_districts = data.nlargest(10, 'total_crime_against_women')
    bar_fig = px.bar(
        top_10_districts, x='total_crime_against_women', y='district_name', orientation='h',
        title='Top 10 Districts by Total Crime Against Women'
    )
    plots['bar'] = json.dumps(bar_fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Generate cluster descriptions
    descriptions = generate_cluster_descriptions(cluster_means)

    return jsonify({'plots': plots, 'descriptions': descriptions})

def generate_cluster_descriptions(cluster_means):
    descriptions = {}
    for cluster, means in cluster_means.iterrows():
        top_crimes = means.nlargest(3)
        description = f"Cluster {cluster} is characterized by high rates of "
        description += ", ".join([f"{crime.replace('_', ' ')} ({value:.2f})" for crime, value in top_crimes.items()])
        descriptions[cluster] = description
    return descriptions

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
