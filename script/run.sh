nohup python src/server/main_service_online.py llm_model >> logs/llm_model.log 2>&1 &
nohup python src/server/main_service_online.py searcher >> logs/searcher.log 2>&1 &