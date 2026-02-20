# semaptic

Semaptic is a utility for Colab to take a CSV, embed the text column with either OpenAI or Gemini embeddings, and then visualize the results in a 2D space with PacMap (or Umap or T-SNE). The processed data with embeddings and coordinates is saved in SQLite format for efficient storage and retrieval.

This helps you explore large text datasets -- you can explore lots of regions on the map and see fewer boring duplicates.

You can try it in action [in this Colab notebook](https://colab.research.google.com/drive/1Y1-lUxXIBpakLyxKu9HUyulqv9IugcYy#scrollTo=kjmcnC-iUsVf)


This is all you need:

```python
from semaptic import embed_reduce_and_map

embed_reduce_and_map(
    "path_to_your_csv.csv", 
    "clean_complaints", 
    keyword_map={ # a few topics and keywords, for orienting yourself in the map
        "politics": ["trump", "biden", "democrat", "republican"], 
        "tech": ["chatgpt", "google", "facebook"],
    })
```

That's equivalent to this:

```python
from semaptic import *
input_filename = "path_to_your_csv.csv"
text_column_name = "clean_complaints" # or whatever the text column is 
model_to_use=MODEL_TO_USE # {"gemini", "openai"}
dim_red_method = "pacmap" # or "umap" or "tsne"

output_filenames  = make_output_filenames(input_filename, dim_red_method)
df = embed_if_necessary(input_filename, text_column_name, model_to_use=model_to_use, dim_red_method=dim_red_method)
tokenize(df)
keyword_map = { # a few topics and keywords, for orienting yourself in the map
        "politics": ["trump", "biden", "democrat", "republican"], 
        "tech": ["chatgpt", "google", "facebook"],
}
topic_classifications(df, keyword_map=keyword_map)
df_pacmap = do_pacmap(df.copy(), output_filenames[model_to_use]["xy"])
plot(df_pacmap, dim_red_method=dim_red_method)
```

## Demo Image

Here is a screenshot of a bunch of FTC complaints about a credit monitoring company, visualized with PacMAP.

![an image of a map with a lot of clusters.](docs/semaptic.png)

You can play with [a live demo](https://jeremybmerrill.github.io/semaptic/live_demo/semaptic.html) using data from some federal government AI use cases.

## TODO list

- [ ] should the embed_reduce_and_map function print itself out, so users can modify it (e.g. to do some light data cleaning on their text column)?
- [ ] use the Gemini embedding types (clustering, semantic search, etc.) and see if that means we get better results.
- [ ] add instructions to the default colab.