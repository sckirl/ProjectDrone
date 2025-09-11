import roboflow

rf = roboflow.Roboflow(api_key="YOUR KEY HERE")
model = rf.workspace().project("people-detection-o4rdr/10").version("1").model
prediction = model.download()