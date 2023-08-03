docker build -t sisap23/julia-example .
docker run -v /home/sisap23evaluation/data:/data:ro -v ./result:/result --stop-timeout 10 -it sisap23/julia-example $1 
