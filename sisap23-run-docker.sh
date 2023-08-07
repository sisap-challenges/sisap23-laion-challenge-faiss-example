docker build -t sisap23/faiss-example .
docker run -v /home/sisap23evaluation/data:/data:ro -v ./result:/result --stop-timeout 10 -it sisap23/faiss-example $1 
