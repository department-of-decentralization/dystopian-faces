gcloud functions deploy dystopian-faces-test --gen2 --runtime=python312 --source=. --entry-point=process_image --trigger-http --memory=512MB