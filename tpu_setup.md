# Set up Grover on TPUs

First, you need to have a Google Cloud Platform account. Then you need to set up the configurations:

1. Open a Cloud Shell window. Create a variable for the project's ID:

```
export PROJECT_ID=project-id
```

2. Configure `gcloud` command-line tool to use the project where you want to create Cloud TPU.

```
gcloud config set project $PROJECT_ID
```

3. Create a Cloud Storage bucket using the following command:

```
gsutil mb -p ${PROJECT_ID} -c standard -l us-central1 -b on gs://bucket-name
```

4. Launch a Compute Engine VM and Cloud TPU using the `gcloud` command.

```
gcloud compute tpus execution-groups create \
 --name=grover \
 --zone=us-central1-f \
 --tf-version=2.5.1 \
 --machine-type=n1-standard-1 \
 --accelerator-type=v2-8
```

5. ssh to your VM. You need to set up the paths in the python script. Make sure the path refers to the correct google cloud bucket.

# Tips
- Make sure to stop your TPU after you're done using it or Google will delete it
- Make sure files you access when using TPU are in cloud storage
