# Data-Science-Assignment

# Test Details:
  This test was created by Aidetic Software Pvt. Ltd. for Data Science Hiring. Anyone can take this test.
  
  Instructions:
  
    1. Fork this repository.
    
    2. Write your code and commit it to your personal fork.
    
    3. Important: Keep you fork private and share it with the following Email IDs:
    
            a) abhishek@aidetic.in
            
            b) yadvendra@aidetic.in
            
            c) ketan@aidetic.in
            
            d) mehul@aidetic.in
            
    4. Fill this form: https://forms.gle/H3SbJ7CfU9bgw8gH8
    
    5. Once you have submitted the form someone from our time will reach out to you for the next steps.

# Overview:
  The goal of this assignment is to develop a noun-chunk based keyword extraction pipeline from
  unstructured text data. The final deliverable is a REST API that takes in a batch of 20 news
  articles as input and returns the top 10 noun chunks encountered in this batch of articles.


# API:
  You can develop this REST API in Flask and deploy it on your local machine to test via Postman.
  The request and response of the API will be in JSON format. Below are details on the format of
  the API and will help you structure it better:

  `Route`: nc_keyword_extraction
  
  `Request`: {"data": ["News Article 1", "News Article 2", "News Article 3" ... "News Article 20"]}
  
  `Response`: {"noun_chunks": {"nc": "Trump Administration", "COVID Vaccine", ... "Crypto trading"}}

# Processing Pipeline:
  To power the API, you'll need to develop a core module that processes the news articles data
  and finds the top 10 noun chunks you need to return to the Flask server. It will have the following
  general components:
    
    1. Noun Chunk extraction engine: You'll need to write a method for finding noun chunks in
    each article. You only need to focus on bigrams and trigrams for now. Unigrams and
    >3-grams can be left out.

    2. Finding top noun chunks: Here, ideally a model (TFIDF/CV) is best suited to find the best
    noun chunks among all the ones extracted in each article. Brownie points for innovating
    with models or approaches  in this section.
    
    3. Eliminating bad noun chunks: Sometimes, you'll get almost duplicate noun chunks or
    ones that do not make sense. You'll need a small post-processing engine to discard or
    deduplicate noun chunks.

Once you've done all this, you should have a pipeline that is able to process a batch of news
articles and return the top noun chunks.

`Bonus Points`: Think about how NER can be used to increase relevance of noun chunks. If your
approach incorporates NER in a smart way, you get bonus points.

# Data
  In order to train any models that you might want for finding top noun chunks or to test your API
  on the batch of news articles, here's some sample data you can use:
  
  https://webhose.io/free-datasets/technology-news-articles/

# Deliverables
  
  1. The final deliverable is a GitHub repository containing all files for the processing pipeline,
  models, Flask server, and a test script. The GitHub repository should be structured as:
  
      ○ src
      
          ■ nc_extraction.py*

          ■ nc_extraction_handler.py (optional)

          ■ server.py*

          ■ utils.py (optional)

          ■ test_request.py (optional)
        
      ○ models
      
          ■ top_nc_model.pkl
        
      ○ requirements.txt
      
      ○ ReadME.md
  
  2. *Do not upload any data or CSV file into GitHub
  
  3. The other deliverable you need to include is a screenshot of the working REST API in
  Postman on your local machine or a VM. If that is not possible, do definitely include a
  test_request.py script in your repository to test the API locally via Python.
  
  4. Coding Instructions and General Guidelines:


      ○ Use object oriented programming. Load your models into memory when you
      instantiate the class and write methods for the different steps in your pipeline.
      
      ○ Write docstrings for your methods.
      
      ○ Format the code properly using a formatter in your IDE.
      
      ○ Explicitly list the data types of your method input and output in your code.
      
      ○ Assign variable names intelligibly.
      
      ○ Adhere to the outlined repository structure as closely as possible.

All the best and do reach out in case you have any questions. Cheers!
