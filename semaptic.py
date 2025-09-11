# -*- coding: utf-8 -*-
"""semantic embeddings mapper (with click-to-select!)

This file is meant to be run from Colab.

"""

"""
!pip install -q pacmap jupyter_dash itables
!wget -q -nc https://filedn.com/lVaAxkskVxILBoUDG3XUrm7/{filename_slug}.csv
# !wget -q -nc https://filedn.com/lVaAxkskVxILBoUDG3XUrm7/{filename_slug}_with_openai_embeddings.csv
# !wget -q -nc https://filedn.com/lVaAxkskVxILBoUDG3XUrm7/{filename_slug}_with_gemini_embeddings.csv
MODEL_TO_USE = "gemini"
assert MODEL_TO_USE in ["openai", "gemini"]
"""

import os
from time import sleep
from ast import literal_eval

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
tqdm.pandas()
from openai import OpenAI
from google import genai
from pacmap import PaCMAP
from sklearn.preprocessing import normalize
from dash import dcc, html, Input, Output, dash_table, Dash
import itables
from itables import init_notebook_mode
import plotly.express as px

DEFAULT_MODEL_TO_USE = "gemini"

try:
  from google.colab import userdata, files
except ImportError:
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

def make_output_filenames(filename): 
  filename_slug = filename.replace(".csv", "").replace("_with_openai_embeddings", "").replace("_with_gemini_embeddings", "").replace("_with_xy", "")
  openai_emb_file = f"{filename_slug}_with_openai_embeddings.csv"
  gemini_emb_file = f"{filename_slug}_with_gemini_embeddings.csv"
  gemini_emb_xy_file = f"{filename_slug}_with_gemini_embeddings_with_xy.csv"
  openai_emb_xy_file = f"{filename_slug}_with_openai_embeddings_with_xy.csv"
  output_filenames = {"openai": {"xy": openai_emb_xy_file, "no_xy": openai_emb_file}, "gemini": {"xy": gemini_emb_xy_file, "no_xy": gemini_emb_file}}
  return output_filenames

def embed_if_necessary(input_filename, text_column_name, model_to_use=DEFAULT_MODEL_TO_USE):
  # prompt: create an embeddings column with the output of the text-embedding-large model from openai
  output_filenames  = make_output_filenames(input_filename)
  if (os.path.exists(output_filenames[model_to_use]["xy"]) or os.path.exists(output_filenames[model_to_use]["no_xy"])):
    raw_df = pd.read_csv(output_filenames[model_to_use]["xy"]) if os.path.exists(output_filenames[model_to_use]["xy"]) else pd.read_csv(output_filenames[model_to_use]["no_xy"]) # TODO: handle xy version
    raw_df["embedding"] = raw_df.embedding.progress_apply(lambda emb: literal_eval(emb) if not pd.isnull(emb) else emb)
  else:
    print("embedding isn't expected, but okay if it is")
    input()
    raw_df = pd.read_csv(input_filename)
    raw_df.rename({text_column_name: "text"}, axis=1, inplace=True)
    raw_df["text_to_embed"] = raw_df.text.str.replace(r"https://[^ ]+", '', regex=True).str.strip()

    if model_to_use == "openai":
      client = OpenAI(api_key = userdata.get('OPENAI_API_KEY'))
      def create_embedding(text):
        if not text or pd.isnull(text) or text == '':
          return None
        response = client.embeddings.create(
            input = text,
            model = "text-embedding-3-large",
        )
        return response.data[0].embedding

      raw_df['embedding'] = raw_df['text_to_embed'].progress_apply(create_embedding)
    elif model_to_use == "gemini":
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
      embed_df['embedding'] = flatten([create_embedding(batch.to_list()) for batch in tqdm(np.array_split(embed_df['text_to_embed'], (len(embed_df) // 100) + 1 ))])
      raw_df.loc[embed_df.index, 'embedding'] = embed_df['embedding']
      output_filename = output_filenames[model_to_use]["no_xy"]
      raw_df.to_csv(output_filename)
      files.download(output_filename)
  return raw_df

def do_pacmap(raw_df, output_filename):
  df = raw_df.dropna(subset=['embedding'])

  embedding_array = np.array(df.embedding.to_list())
  normed_truth_embeddings = normalize(embedding_array, norm='l2')
  pacmap = PaCMAP(random_state=0, n_components=2, n_neighbors=None)
  pacmap_embedding = pacmap.fit_transform(normed_truth_embeddings)

  df["x"] = pacmap_embedding[:, 0]
  df["y"] = pacmap_embedding[:, 1]

  df.to_csv(output_filename)
  files.download(output_filename)

  return df

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


def plot(df, what_to_display="term_frequencies"):
  assert what_to_display in ["term_frequencies", "text_counts", "topic_counts"]
  df["display_text"] = df.text.str.replace(r"https://[^ ]+", '', regex=True).str.wrap(100).str.replace("\n", "<br />")
  fig = px.scatter(df, x="x", y="y", color="topic", hover_name="display_text",
                    title='PaCMAP projection of ChatGPT chat titles',
                    opacity=0.4,
                    width=1200, height=800)

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
  app.run(mode='inline'); # This will embed the app directly in Colab

  # this cell is meant to be interactive, shown only when you interact with the plot above
  init_notebook_mode(all_interactive=True)
  if len(selected_df):
    with pd.option_context('display.max_colwidth', None, 'display.max_rows', 500):
      itables.show(selected_df[["created_at", "text", "url"]])



def do_everything(input_filename, text_column_name, keyword_map={}, model_to_use=DEFAULT_MODEL_TO_USE, what_to_display="term_frequencies"):
  output_filenames  = make_output_filenames(input_filename)
  df = embed_if_necessary(input_filename, text_column_name, model_to_use=model_to_use)
  tokenize(df)
  topic_classifications(df, keyword_map=keyword_map)
  df = do_pacmap(df, output_filenames[model_to_use]["xy"])
  plot(df, what_to_display=what_to_display)