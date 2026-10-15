# Lindah's VAE notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfk1327/ai_weather_pres3/blob/main/people/lindah/vae_minimal.ipynb)

**`vae_minimal.ipynb`** trains a small VAE on 40 years of daily-max wind maps over Ireland, then
generates new (and more extreme) maps. It's your own copy, so edit and break whatever you like.

## Run it on Colab
1. Click the **Open in Colab** badge above.
2. **Runtime → Change runtime type → T4 GPU → Save.**
3. **Runtime → Run all.** The first cell downloads the code and data (~35 MB); training takes about a minute.
4. The saved version on GitHub already shows the outputs, so you can see what to expect before you run it.

## Experiment
- Change the numbers in **Section 1 (settings)**, then **Runtime → Run all** again.
- **Section 10** has a list of things to try, including breaking it on purpose.
- Change one thing at a time and write down what happened.

## Saving your work
- **File → Save a copy in Drive** keeps your edited notebook in your Google Drive.
- To put it back in this repo (so the team can see it): **File → Save a copy in GitHub**, pick
  `jfk1327/ai_weather_pres3`, path `people/lindah/vae_minimal.ipynb`. Needs you to be added as a
  collaborator on the repo (ask James).
- Broke it beyond repair? Click the Colab badge again for a fresh copy.

## Run it on a laptop instead
From the repo root: `pip install -r requirements.txt`, then open the notebook in Jupyter or VS Code.
