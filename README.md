# Faulty Machine State classifier

- **animall-assignment.ipnyb** contains different models for Faulty Machine State classification

- **code/** folder contains the best performing model - BalancedRandomForestClassifier : Here the code for building fastAPI endpoint and docker deployment resides 

### Commands 

- To test the code locally without deployment 
 
 `python .code/main.py`
 
 - To deploy the code using docker compose 
 
 `docker-compose up --build`
 
Once the application startup is complete, you can test the APIs at http://127.0.0.1:8000/docs
