# Text_classfication_spacy_Google_autoML

## Ojective
The goals of this posting are 
- provide executable codes to do text classification using open sources spacy package. 
- user experince of the Google autoML natural language which was realsed by the end of last year.

The link of [google autoML](https://cloud.google.com/natural-language/#how-automl-natural-language-works)
The googel autoML shows great advantage than the open sources spacy in terms of user-friendly UI, prediction accuracy, model depolyment and time saving. The only drawback is that the uploading data, preprocessing  and model traning of Google autoML is very slow. It takes 5~7 hours to finish a training of the small dataset attached here.

##Package install

    $ conda install -c conda-forge spacy
    $ python -m spacy download en_core_web_sm
    
![Alt text](Google_auto_ML.png?raw=true "Optional Title")    
