# semaptic

Semaptic is a utility for Colab to take a CSV, embed the text column with either OpenAI or Gemini embeddings, and then visualize the results in a 2D space with PacMap.

This helps people explore large text datasets.

You can try it in action here: https://colab.research.google.com/drive/1Y1-lUxXIBpakLyxKu9HUyulqv9IugcYy#scrollTo=kjmcnC-iUsVf

## TODO list

- [ ] should the do_everything function print itself out, so users can modify it (e.g. to do some light data cleaning on their text column)?
- [ ] use the Gemini embedding types (clustering, semantic search, etc.) and see if that means we get better results.
- [ ] add instructions to the default colab.
- [ ] add a (small) rendered map to this README.