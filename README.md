# About Project Structure
1) the data directory consists of the decomposed data obtained by passing the original data through the table-decomposition pipeline.
2) The models consists the code for fine tuning different models (T5, BART, OmniTab) and LLMS (GPT, Claude, Llama)
3) We used fact extraction technique to provide the model with the extract facts based on user query and have utilized table question answering model TAPAS for this task.
4) The table-decomposition directory consists the code for LLM based table decomposition which takes in original data and keeps only relevant data based on user query. 


# To read more
[https://drive.google.com/file/d/1mKiSvr2W50X6MuujFkG1oW09lHI-941N/view?usp=sharing](https://www.dropbox.com/scl/fi/g64w2vqi3pvtscaoeri0w/DETQUS.pdf?rlkey=nx3r96fafccaskxew3yv50m9h&st=nwnmhd7q&dl=0)
