# Deploy the Translator on the cloud

Here we deploy the trained German-English translator onto AWS EC2 instance. Here are the instructions. 

## Create a flask app 
In [app.py](app.py) flask is used to create api endpoints services. Two endpoints are created: 
* "@app.route('/')". Use GET to access this endpoint and it will return "Hello World!". 
* "@app.route('/predict', methods=['POST'])" Use POST with a German sentence the endpoint will return the translated English sentence. 

## Launch a AWS EC2 instance
Select Amazon Linux 2 AMI. Since loading the translator model requires more than 1G of memory. Here we select the t2.small machine which has 2G of memory. In this step it requires to setup the key-pairs and IAM settings. 

## Install docker
ssh to EC2 instance. Then run the following command to install docker. 
* sudo yum update -y 
* sudo amazon-linux-extras install docker 
* sudo service docker start (or sudo systemctl enable docker) 
* sudo docker info 

## Upload files 
scp the following files and folder to EC2
* [app.py](app.py) (the flask app)
* translator/ (the trained translator model)
* [requirements.txt](requirements.txt) (the python library requirements)
* [Dockerfile](Dockerfile) (docker build script)

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
