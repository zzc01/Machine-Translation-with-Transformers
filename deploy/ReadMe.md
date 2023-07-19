# Deploy the Translator on the cloud

Here we deploy the trained german-english translator onto AWS EC2 instance. Here are the instructions. 

## Launch a AWS EC2 instance
Select Amazon Linux 2 AMI. Because loaingd the translator model requires more than 1G of memory. Therefore, select the t2.small machine which has 2G of memory. Then setup the key-pairs and IAM settings. 

## Install docker
ssh to EC2 instance. Then run the following command to install docker. 
* sudo yum update -y 
* sudo amazon-linux-extras install docker 
* sudo service docker start (or sudo systemctl enable docker) 
* sudo docker info 

## Upload files 
scp the following files and folder to EC2
* app.py (the flask app)
* translator/ (the trained translator model)
* requirements.txt (the python library requirements)
* Dockerfile (docker build script)

## Bring up the docker container 
Use the following command to buid the docker image and run the container: 
* sudo docker build -t webserver .
* sudo docker images -a
* sudo docker run -p 80:5000 webserver .

## Try out the German-English translator
Open the jupyter notebook [webservice_api.ipynb](webservice_api.ipynb) and follow the script to test the translator! 

<pre><p align="center">
<img src="https://github.com/zzc01/Machine-Translation-with-Transformers/assets/86133411/530d5c17-8a83-4935-b0ee-2ca749f4bb18"  width="400" >
</p></pre>

# References 
[1] [Simple Way to Deploy Machine Learning Models to Cloud](https://towardsdatascience.com/simple-way-to-deploy-machine-learning-models-to-cloud-fd58b771fdcf) <br/>
