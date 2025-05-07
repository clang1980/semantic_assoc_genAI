This repo contains the data the paper "Using LLMs for experimental stimulus pretests in linguistics. Evidence from semantic associations between words and social gender" for Konvens 2025 is based on.

- "noun_list_uni.csv" contains our dataset, the list of German NPs 
- the directory "data" contains the aggregated LLM and human annotator data needed for further analysis
- "combine_datasets_visualizations.R" is the R script used for data analysis and visualization
- the directory "results" contains both the visualizations presented in the paper and additional visualizations which did not fit into the paper for reasons of limited space
- the directory "vote_viewer_app" contains the code and the data for the interactive visualization mentioned in the paper

- the directory "sample_workflow" contains python scripts that were used to generate, extract and process LLama 3.1 70B data (ID tokens have been removed and the workflow of the other LLMs differs in minor details)


