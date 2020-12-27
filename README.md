# CNN on CIFAR-10 with Horovod

## How to compute on Plafrim

 `*0 - Lancez un run avec salloc et connectez vous sur votre noeud (1 noeud et quelques cpu suffisent) : salloc -p mistral -N 1 --mincpu=20 --time=6:00:00`
 
 `*1 - activez anaconda : source ~/anaconda/bin/activate`
 
 `*2 - Créez un environnement dédié au tp : conda create -y --name horovod python=3.8 pip`
 
 `*3 - activez l'environnement : conda activate horovod`
 
 `*4 - installez les modules nécessaires au TP : pip install -r requirements.txt `
 
 `*5 - installez tensorflow sur l'environnement virutel : conda install tensorflow`

 `*6 - installez keras sur l'environnement virutel : conda install keras`
 
 `*7 - lancez jupyter-lab : jupyter-lab --ip=0.0.0.0`
 
 `*8 - identifiez le port sur lequel il s'est lancé (normalement 8888) ou précisez --port 8888`
 
 `*9 - depuis votre machine locale faites un port forwarding pour accéder au port (8888) de votre noeud : ssh -L 8888:mistralXX:8888 formation (Attention à bien remplacer mistralXX par la machine sur laquelle vous avez allouer les noeuds)`
 

