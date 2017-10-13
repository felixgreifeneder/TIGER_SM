# TIGER_SM
Sentinel-1 soil moisture mapping

This script is intended to be used within the processing plugin of QGIS. It allows the create an estimated soil moisture map utilising Copernicus Sentinel-1 backscatter measurements. The retrieval of soil moisture is based on a Support-Vector-Regression machine learning approach. The model training was performed, based on in-situ data collected within several field campaigns in Kenya. The development of this software and the collection of in-situ data was supported by funding of European Space Agency project framework "TIGER Bridge" within the project "Synergy of multi-temporal Sentinel-1 and Sentinel-2 image for soil water resources monitoring in Africa". For details on the applied approach, please see: 

Pasolli, L., C. Notarnicola, G. Bertoldi, L. Bruzzone, R. Remelgado, F. Greifeneder, G. Niedrist, S. Della Chiesa, U. Tappeiner, and M. Zebisch. 2015. Estimation of Soil Moisture in Mountain Areas Using SVR Technique Applied to Multiscale Active Radar Images at C-Band. Sel. Top. Appl. Earth Obs. Remote Sens. 8(1): 262–283.

Greifeneder, F., E. Khamala, D. Sendabo, M. Zebisch, H. Farah, and C. Notarnicola. 2017. Mapping Soil Moisture Content Using Sentinel-1 and Sentinel-2: Case Studies From Kenya. p. 263. In ISRSE-37. Tshwane, South Africa.

Credit: Based on modified Copernicus Sentinel data (2015-2017)/ESA

Please follow the installation instructions below:

<h2> Installation </h2>
Most of the data processing is executed on-line on Google Earth Engine. Therefore, the execution of this script requires a Google account and access to Google Earth Engine - we are working on an updated version that will overcome this requirement.

<h3> Installation of the Google Earh Engine API </h3>
To allow the script to talk to Google Earh Engine the API has to be installed. Please follow the instructions at this link: <a href="https://developers.google.com/earth-engine/python_install_manual">GEE API</a>
<h3>Installtion of the Google Drive API</h3>
After the computation inside Google Earth Engine is finished, the results are exported to your Google Drive. To let the script access and download the results to you local computer, the Google Drive API has to be installed as well. Please follow the instructions here:
<a href="https://developers.google.com/drive/v3/web/quickstart/python">Google Drive API</a> \n
As described in the manual, for the first run, the authentication can be initiated by running the quickstart.py script. To enable the download of data please modify the following line of the script: 
SCOPES = 'https://www.googleapis.com/auth/drive.metadata.readonly' --> SCOPES = 'https://www.googleapis.com/auth/drive'
<h3>Required Python Packages</h3>
If executed under Windows, the script is compatible with the OSGeo4W Python distribution. The following Python libraries must be installed: \n
- google-api-python-client (see above) \n
- earthengine-api (see above) \n
- numpy \n
- scipy \n
- scikit-learn \n
- matplotlib
- httplib2
