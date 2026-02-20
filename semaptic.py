# -*- coding: utf-8 -*-
"""semantic embeddings mapper (with click-to-select!)
"""

"""
!pip install -q pacmap jupyter_dash itables
!wget -q -nc https://filedn.com/lVaAxkskVxILBoUDG3XUrm7/{filename_slug}.csv
# !wget -q -nc https://filedn.com/lVaAxkskVxILBoUDG3XUrm7/{filename_slug}_with_openai_embeddings.db
# !wget -q -nc https://filedn.com/lVaAxkskVxILBoUDG3XUrm7/{filename_slug}_with_gemini_embeddings.db
MODEL_TO_USE = "gemini"
assert MODEL_TO_USE in ["openai", "gemini"]
"""

import os
import sqlite3
from time import sleep
from ast import literal_eval
from concurrent.futures import ThreadPoolExecutor, as_completed

# on some platforms (like Jeremy's computer), this leads to a difficult-to-debug
# error like "ValueError: could not broadcast input array from shape (0,) into shape (64,)"
# this env var fixes
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
tqdm.pandas()
from openai import OpenAI
from google import genai
from pacmap import PaCMAP
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
import umap
from dash import dcc, html, Input, Output, dash_table, Dash
import itables
from itables import init_notebook_mode
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.io import renderers
renderers.default = "plotly_mimetype+notebook_connected"

DEFAULT_MODEL_TO_USE = "gemini"

def save_df_to_sqlite(df, db_filename, table_name="data"):
  """Save dataframe to SQLite database"""
  # Convert any list/array columns to strings for storage
  df_copy = df.copy()
  
  # Convert all object columns that contain lists to strings
  for col in df_copy.columns:
    if df_copy[col].dtype == 'object':
      # Check first non-null value to see if it's a list
      first_non_null = df_copy[col].dropna().iloc[0] if not df_copy[col].dropna().empty else None
      if first_non_null is not None and isinstance(first_non_null, (list, np.ndarray)):
        print(f"Converting list column '{col}' to string format")
        df_copy[col] = df_copy[col].apply(lambda x: str(x) if x is not None else None)
  
  with sqlite3.connect(db_filename) as conn:
    df_copy.to_sql(table_name, conn, if_exists='replace', index=True)

def load_df_from_sqlite(db_filename, table_name="data"):
  """Load dataframe from SQLite database"""
  with sqlite3.connect(db_filename) as conn:
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn, index_col='index')
  
  # Convert string representations back to lists for specific columns
  list_columns = ['embedding', 'tokens']  # Add other list columns as needed
  for col in list_columns:
    if col in df.columns:
      def safe_literal_eval(x):
        if pd.isnull(x) or x == 'nan' or x == 'None' or str(x).lower() == 'nan':
          return None
        try:
          return literal_eval(str(x))
        except (ValueError, SyntaxError) as e:
          # For debugging, let's see what values are causing issues
          if len(str(x)) < 200:  # Only print short problematic values
            print(f"Warning: Could not parse {col} value: {str(x)} Error: {e}")
          return None
      
      df[col] = df[col].progress_apply(safe_literal_eval)
  
  return df

def download_sqlite_file(filename):
  """Download SQLite file if running in Colab"""
  files.download(filename)

try:
  from google.colab import userdata, files, output as colab_output
  in_colab = True
except ImportError:
   # if we're not in colab, make stubs
   class colab_output:
     @staticmethod
     def serve_kernel_port_as_iframe(port):
        # no-op
        pass
   print("not running in colab, so userdata and files modules not available") 
   # make a stub files with a download function that does nothing
   class files:
     @staticmethod
     def download(filename):
       print(f"files.download({filename}) called, but not running in colab so not doing anything")
   userdata = {
       'OPENAI_API_KEY': os.getenv("OPENAI_API_KEY"),   
        'GEMINI_API_KEY': os.getenv("GEMINI_API_KEY"),
    }
   in_colab = False

def make_output_filenames(filename, dim_red_method="pacmap"): 
  filename_slug = filename.replace(".csv", "").replace("_with_openai_embeddings", "").replace("_with_gemini_embeddings", "").replace("_with_xy", "")
  openai_emb_file = f"{filename_slug}_with_openai_embeddings.db"
  gemini_emb_file = f"{filename_slug}_with_gemini_embeddings.db"
  gemini_emb_xy_file = f"{filename_slug}_with_gemini_embeddings_with_{dim_red_method}_xy.db"
  openai_emb_xy_file = f"{filename_slug}_with_openai_embeddings_with_{dim_red_method}_xy.db"
  output_filenames = {"openai": {"xy": openai_emb_xy_file, "no_xy": openai_emb_file}, "gemini": {"xy": gemini_emb_xy_file, "no_xy": gemini_emb_file}}
  return output_filenames

def _parallel_process(items, worker_fn, max_workers=10, desc="Processing"):
  """
  Generic parallel processing helper using ThreadPoolExecutor.
  
  Args:
    items: List of (index, data) tuples to process
    worker_fn: Function that takes data and returns result
    max_workers: Number of parallel workers
    desc: Description for progress bar
    
  Returns:
    List of results in the same order as items
  """
  results = [None] * len(items)
  
  with ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_idx = {executor.submit(worker_fn, data): idx for idx, data in items}
    for future in tqdm(as_completed(future_to_idx), total=len(items), desc=desc):
      idx = future_to_idx[future]
      results[idx] = future.result()
  
  return results

def _create_openai_embeddings(raw_df):
  """Create embeddings using OpenAI API with parallel processing."""
  client = OpenAI(api_key = userdata.get('OPENAI_API_KEY'))
  
  def create_embedding(text):
    if not text or pd.isnull(text) or text == '':
      return None
    response = client.embeddings.create(
        input = text,
        model = "text-embedding-3-large",
    )
    return response.data[0].embedding
  
  texts_to_embed = list(enumerate(raw_df['text_to_embed']))
  embeddings = _parallel_process(texts_to_embed, create_embedding, desc="Creating OpenAI embeddings")
  raw_df['embedding'] = embeddings

def _create_gemini_embeddings(raw_df):
  """Create embeddings using Gemini API with batch processing."""
  client = genai.Client(api_key=userdata.get("GEMINI_API_KEY"))
  
  def flatten(xss):
    return [x for xs in xss for x in xs]
  
  def create_embedding(texts):
    response = client.models.embed_content(
            model="text-embedding-004",
            contents=texts,
            config=genai.types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
            )
    sleep(2.5) # 1,500 requests per minute https://ai.google.dev/gemini-api/docs/models#text-embedding
    return [emb.values for emb in response.embeddings]
  
  embed_df = raw_df[~raw_df["text_to_embed"].isna() & raw_df["text_to_embed"].str.len() > 0].copy()
  embed_df['embedding'] = flatten([create_embedding(batch.to_list()) for batch in tqdm(np.array_split(embed_df['text_to_embed'], (len(embed_df) // 100) + 1))])
  raw_df.loc[embed_df.index, 'embedding'] = embed_df['embedding']

def embed_df(raw_df, text_column_name, model_to_use=DEFAULT_MODEL_TO_USE):
  raw_df.rename({text_column_name: "text"}, axis=1, inplace=True)
  raw_df["text_to_embed"] = raw_df.text.str.replace(r"https://[^ ]+", '', regex=True).str.strip()  
  if model_to_use == "openai":
    _create_openai_embeddings(raw_df)
  elif model_to_use == "gemini":
    _create_gemini_embeddings(raw_df)
  return raw_df 

def embed_if_necessary(input_filename, text_column_name, model_to_use=DEFAULT_MODEL_TO_USE, dim_red_method="pacmap"):
  # prompt: create an embeddings column with the output of the text-embedding-large model from openai
  output_filenames  = make_output_filenames(input_filename, dim_red_method)
  
  db_filename = None
  # Prefer the xy file if it exists and has data, otherwise use no_xy file
  if os.path.exists(output_filenames[model_to_use]["xy"]):
    with sqlite3.connect(output_filenames[model_to_use]["xy"]) as conn:
      try:
        result = pd.read_sql_query("SELECT COUNT(*) as count FROM data", conn)
        if result['count'].iloc[0] > 0:
          db_filename = output_filenames[model_to_use]["xy"]
      except:
        pass  # Table doesn't exist or other error
  
  # If xy file is empty or doesn't exist, check no_xy file
  if db_filename is None and os.path.exists(output_filenames[model_to_use]["no_xy"]):
    with sqlite3.connect(output_filenames[model_to_use]["no_xy"]) as conn:
      try:
        result = pd.read_sql_query("SELECT COUNT(*) as count FROM data", conn)
        if result['count'].iloc[0] > 0:
          db_filename = output_filenames[model_to_use]["no_xy"]
      except:
        pass  # Table doesn't exist or other error
  
  if db_filename is not None:
    raw_df = load_df_from_sqlite(db_filename)
  else:
    print("embedding isn't expected, but okay if it is")
    input()
    raw_df = pd.read_csv(input_filename)
    raw_df = embed_df(raw_df, text_column_name=text_column_name, model_to_use=model_to_use)
    output_filename = output_filenames[model_to_use]["no_xy"]
    save_df_to_sqlite(raw_df, output_filename)
    download_sqlite_file(output_filename)
  return raw_df

def _do_dimensionality_reduction(raw_df, method_name, reducer, output_filename=None):
  """Generic dimensionality reduction function"""
  df = raw_df.dropna(subset=['embedding'])

  embedding_array = np.array(df.embedding.to_list())
  normed_truth_embeddings = normalize(embedding_array, norm='l2')
  embedding_2d = reducer.fit_transform(normed_truth_embeddings)

  raw_df.loc[df.index, f"x_{method_name}"] = embedding_2d[:, 0]
  raw_df.loc[df.index, f"y_{method_name}"] = embedding_2d[:, 1]

  save_df_to_sqlite(raw_df, output_filename)
  download_sqlite_file(output_filename)

  return raw_df

def do_pacmap(raw_df, output_filename=None):
  """
  add columns x_pacmap, y_pacmap for PacMAP's reduction of the `embedding` field in `raw_df` to 2D

  raw_df: a pandas DataFrame with an `embedding` column
  output_filename [optional]: write `raw_df` to disk as sqlite3 at this filename, if specified
                              if on colab, try to download the sqlite too.
  """
  pacmap = PaCMAP(random_state=0, n_components=2, n_neighbors=None)
  return _do_dimensionality_reduction(raw_df, "pacmap", pacmap, output_filename)

def do_umap(raw_df, output_filename=None):
  """
  add columns x_umap, y_umap for UMAP's reduction of the `embedding` field in `raw_df` to 2D

  raw_df: a pandas DataFrame with an `embedding` column
  output_filename [optional]: write `raw_df` to disk as sqlite3 at this filename, if specified
                              if on colab, try to download the sqlite too.
  """
  umap_reducer = umap.UMAP(random_state=0, n_components=2)
  return _do_dimensionality_reduction(raw_df, "umap", umap_reducer, output_filename)

def do_tsne(raw_df, output_filename=None):
  """
  add columns x_tsne, y_tsne for t-SNE's reduction of the `embedding` field in `raw_df` to 2D

  raw_df: a pandas DataFrame with an `embedding` column
  output_filename [optional]: write `raw_df` to disk as sqlite3 at this filename, if specified
                              if on colab, try to download the sqlite too.
  """
  df = raw_df.dropna(subset=['embedding'])
  tsne = TSNE(random_state=0, n_components=2, perplexity=min(30, len(df)-1))
  return _do_dimensionality_reduction(raw_df, "tsne", tsne, output_filename)

def do_pacmap_3d(raw_df, output_filename=None):
  """
  add columns x_pacmap_3d, y_pacmap_3d, z_pacmap_3d for PacMAP's reduction of the `embedding` field in `raw_df` to 3D

  raw_df: a pandas DataFrame with an `embedding` column
  output_filename [optional]: write `raw_df` to disk as sqlite3 at this filename, if specified
                              if on colab, try to download the sqlite too.
  """
  df = raw_df.dropna(subset=['embedding'])

  embedding_array = np.array(df.embedding.to_list())
  normed_truth_embeddings = normalize(embedding_array, norm='l2')
  pacmap = PaCMAP(random_state=0, n_components=3, n_neighbors=None)
  embedding_3d = pacmap.fit_transform(normed_truth_embeddings)

  raw_df.loc[df.index, "x_pacmap_3d"] = embedding_3d[:, 0]
  raw_df.loc[df.index, "y_pacmap_3d"] = embedding_3d[:, 1]
  raw_df.loc[df.index, "z_pacmap_3d"] = embedding_3d[:, 2]

  save_df_to_sqlite(raw_df, output_filename)
  download_sqlite_file(output_filename)

  return raw_df

def topic_classifications(df, keyword_map):
  """
  df: dataframe with a `text` column
  keyword_map: dict of topic -> list of keywords, case-insensitively matched to the `text` column in `df`
  
  if you need something fancier, just replicate this:
  
  df.loc[df.text.str.contains('health|cancer|covid', case=False), "topic"] = "health"

  in-place
  """
  df["topic"] = "uncategorized"
  for topic, keywords in keyword_map.items():
    pattern = '|'.join(keywords)
    df.loc[df.text.str.contains(pattern, case=False), "topic"] = topic

def tokenize(df):
  """
  calculate tokens column from text column, removing punctuation and lowercasing

  used for term frequency analysis of selected vs non-selected points

  in-place
  """
  df["tokens"] = df.text.str.replace(r"[^A-Za-z0-9\-]", '').str.lower().str.split()


def calc_term_freqs(df_a, df_b, token_col_a, token_col_b=None, token_min_count_threshold=5, quiet=False):
    """
    given two dataframes with a column for tokenized text, calculate the comparative frequencies of each token

    df_a, df_b: dataframes. Usually actually two slices of one dataframe, e.g. all posts before/after a date, or from user a and user b.
    token_col_a, token_col_b: the name of the token column in each dataframe.
    token_min_count_threshold: the minimum number of times a token must appear in BOTH corpora to be included in the analysis.
    quiet: if True, don't print the number of posts and tokens in each corpus

    output:
    `a_b_freq_ratio`: ratio of token frequency in df_a to token frequency in df_b.
    """
    if not token_col_b:
        token_col_b = token_col_a
    tokens_df_a = pd.DataFrame({"token_count": df_a[token_col_a].explode().value_counts()})
    tokens_df_a["freq"] = tokens_df_a.token_count / tokens_df_a.token_count.sum()
    tokens_df_b = pd.DataFrame({"token_count": df_b[token_col_b].explode().value_counts()})
    tokens_df_b["freq"] = tokens_df_b.token_count / tokens_df_b.token_count.sum()
    if not quiet:
        print("{} A posts, {} B posts; {:,} tokens A, {:,} tokens B".format(
            len(df_a),
            len(df_b),
            sum(tokens_df_a.token_count),
            sum(tokens_df_b.token_count),
        ))

    tokens_df_a = tokens_df_a[tokens_df_a.token_count >= token_min_count_threshold]
    tokens_df_b = tokens_df_b[tokens_df_b.token_count >= token_min_count_threshold]
    joined_tokens_df = tokens_df_a.join(tokens_df_b, rsuffix="_b", lsuffix="_a", how="outer")
    joined_tokens_df["a_b_freq_ratio"] = joined_tokens_df.freq_a / joined_tokens_df.freq_b
    joined_tokens_df = joined_tokens_df.sort_values("a_b_freq_ratio", ascending=False)
    return joined_tokens_df

# To get the selected data, you would typically use a Dash application
# for real-time interaction. However, for a simple Colab notebook
# you can use fig.data[0].selectedpoints after a selection has been made
# (though this requires manual interaction and then running a cell).

# For a more robust, real-time solution in Colab, you'd need Dash:
# (This part isn't directly runnable by itself to show the output in Colab
# without a full Dash server, but it demonstrates the concept.)


def plot(df, what_to_display="term_frequencies", dim_red_method="pacmap", use_dash=True):
  assert what_to_display in ["term_frequencies", "text_counts", "topic_counts"]
  df["display_text"] = df.text.str.replace(r"https://[^ ]+", '', regex=True).str.wrap(100).str.replace("\n", "<br />")
  
  # Handle 3D case with three separate 2D plots
  if dim_red_method == "pacmap_3d":
    # Create subplot figure with 1 row and 3 columns
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('X vs Y', 'X vs Z', 'Y vs Z'),
        horizontal_spacing=0.1
    )
    
    # Create consistent color mapping for topics
    unique_topics = df['topic'].unique()
    colors = px.colors.qualitative.Plotly[:len(unique_topics)]
    color_map = {topic: colors[i] for i, topic in enumerate(unique_topics)}
    
    # Common scatter plot settings
    scatter_kwargs = dict(
        mode='markers',
        marker=dict(opacity=0.4),
        text=df["display_text"],
        hovertemplate='%{text}<extra></extra>'
    )
    
    # Add X vs Y plot
    for topic in unique_topics:
        topic_data = df[df['topic'] == topic]
        fig.add_trace(go.Scatter(
            x=topic_data['x_pacmap_3d'],
            y=topic_data['y_pacmap_3d'],
            name=topic,
            legendgroup=topic,
            marker=dict(color=color_map[topic], opacity=0.4),
            **{k: v for k, v in scatter_kwargs.items() if k != 'marker'}
        ), row=1, col=1)
    
    # Add X vs Z plot
    for topic in unique_topics:
        topic_data = df[df['topic'] == topic]
        fig.add_trace(go.Scatter(
            x=topic_data['x_pacmap_3d'],
            y=topic_data['z_pacmap_3d'],
            name=topic,
            legendgroup=topic,
            showlegend=False,
            marker=dict(color=color_map[topic], opacity=0.4),
            **{k: v for k, v in scatter_kwargs.items() if k != 'marker'}
        ), row=2, col=1)
    
    # Add Y vs Z plot
    for topic in unique_topics:
        topic_data = df[df['topic'] == topic]
        fig.add_trace(go.Scatter(
            x=topic_data['y_pacmap_3d'],
            y=topic_data['z_pacmap_3d'],
            name=topic,
            legendgroup=topic,
            showlegend=False,
            marker=dict(color=color_map[topic], opacity=0.4),
            **{k: v for k, v in scatter_kwargs.items() if k != 'marker'}
        ), row=3, col=1)
    
    fig.update_layout(
        title=f'{dim_red_method.upper()} projection of text data - 2D projections',
        width=600, height=1800
    )
    
  else:
    # Use algorithm-specific column names for 2D plots
    x_col = f"x_{dim_red_method}"
    y_col = f"y_{dim_red_method}"
    
    fig = px.scatter(df, x=x_col, y=y_col, color="topic", hover_name="display_text",
                      title=f'{dim_red_method.upper()} projection of text data',
                      opacity=0.4,
                      width=1200, height=800)

  if use_dash: 
    app = Dash(__name__)

    app.layout = html.Div([
        dcc.Graph(id='scatter-plot', figure=fig),
        html.Div(id='selected-data-output')
    ])

    selected_df = pd.DataFrame()
    @app.callback(
        Output('selected-data-output', 'children'),
        Input('scatter-plot', 'selectedData')
    )
    def display_selected_data(selectedData):
        global selected_df
        # global selected_data # temp
        # global selected_points_indices # temp
        if selectedData:
            selected_data = selectedData

            # gemini generated this, but it doesn't work
            # the pointIndex value doesn't match up with the index in the dataframe, oddly.
            # selected_points_indices = [point['pointIndex'] for point in selectedData['points']]
            # selected_df = df.iloc[selected_points_indices]

            selected_df = df[df.display_text.isin([p['hovertext'] for p in selected_data["points"]])]
            non_selected_df = df[~df.index.isin(selected_df.index)]
            if what_to_display == "text_counts":
              return html.Div([
                  html.H4("Selected Data:" + str(len(selected_df))),
                  dash_table.DataTable(pd.DataFrame(selected_df.display_text.rename("text").value_counts().reset_index()).to_dict('records'), [{"name": i, "id": i} for i in ["text", "count"]])
              ])
            if what_to_display == "topic_counts":
              return html.Div([
                  html.H4("Selected Data:" + str(len(selected_df))),
                  dash_table.DataTable(pd.DataFrame(selected_df.topic.value_counts().reset_index()).to_dict('records'), [{"name": i, "id": i} for i in ["topic", "count"]])
              ])          
            elif what_to_display == "term_frequencies":
              term_freqs = calc_term_freqs(selected_df, non_selected_df, "tokens")
              term_freq_head_and_tail = pd.concat([term_freqs.head(20), term_freqs.sort_values("a_b_freq_ratio", ascending=True).head(20)])
              return html.Div([
                  html.H4("Selected Data:" + str(len(selected_df))),
                  dash_table.DataTable(pd.DataFrame(selected_df.topic.value_counts().reset_index()).to_dict('records'), [{"name": i, "id": i} for i in ["topic", "count"]]),
                  dash_table.DataTable(term_freq_head_and_tail.reset_index().to_dict('records'), [{"name": i, "id": i} for i in ["tokens", "token_count_a","freq_a","token_count_b","freq_b","a_b_freq_ratio"]])
              ])
        return html.Div("No points selected.")

    # To run the Dash app in Colab:
    colab_output.serve_kernel_port_as_iframe(8050)
    app.run(jupyter_mode='inline'); # This will embed the app directly in Colab

    # this cell is meant to be interactive, shown only when you interact with the plot above
    init_notebook_mode(all_interactive=True)
    if len(selected_df):
      with pd.option_context('display.max_colwidth', None, 'display.max_rows', 500):
        itables.show(selected_df[["created_at", "text", "url"]])
  else:
    fig.show(renderer="plotly_mimetype+notebook_connected")


def embed_reduce_and_map(input_filename, text_column_name, keyword_map={}, model_to_use=DEFAULT_MODEL_TO_USE, what_to_display="term_frequencies", dim_red_method="pacmap", use_dash=True):  
  """
  do everything: embed the text, reduce the dimensions, and map it with an interactive plot.

  input_filename: a csv file with a column of text to embed
  text_column_name: the name of the column in the csv file that contains the text to embed
  keyword_map: a dict of topic -> list of keywords, case-insensitively matched to the text column, used for simple topic classification and coloring in the plot
  model_to_use: "openai" or "gemini", which embedding model to use
  what_to_display: what to show in the interactive plot when points are selected. "term_frequencies" will show the tokens that are most overrepresented in the selected region vs the non-selected region. "text_counts" will show the most common texts in the selected region. "topic_counts" will show the most common topics (as defined by the keyword_map) in the selected region. (No-op if use_dash is False)
  dim_red_method: which dimensionality reduction method to use for the plot. "pacmap" is the default and my favorite, but "umap" and "tsne" are also options, and "pacmap_3d" will give you three separate 2D plots for the three dimensions of a 3D PacMAP reduction.
  use_dash: whether to use Dash for the interactive plot. If False, a static Plotly plot will be used. Dash is only interactive in a live notebook (Jupyter, VSCode, Colab all ok); Plotly is still interactive in HTML, but the interactivity is limited to zooming and hovering, not selecting regions and seeing stats about them.
  """
  assert dim_red_method in ["pacmap", "umap", "tsne", "pacmap_3d"], f"dim_red_method must be one of 'pacmap', 'umap', 'tsne', 'pacmap_3d', got {dim_red_method}"
  
  output_filenames = make_output_filenames(input_filename, dim_red_method)
  df = embed_if_necessary(input_filename, text_column_name, model_to_use=model_to_use, dim_red_method=dim_red_method)
  tokenize(df)
  topic_classifications(df, keyword_map=keyword_map)
  
  # Apply the chosen dimensionality reduction method
  if dim_red_method == "pacmap":
    df = do_pacmap(df, output_filenames[model_to_use]["xy"])
  elif dim_red_method == "umap":
    df = do_umap(df, output_filenames[model_to_use]["xy"])
  elif dim_red_method == "tsne":
    df = do_tsne(df, output_filenames[model_to_use]["xy"])
  elif dim_red_method == "pacmap_3d":
    df = do_pacmap_3d(df, output_filenames[model_to_use]["xy"])
  
  plot(df, what_to_display=what_to_display, dim_red_method=dim_red_method, use_dash=use_dash)
