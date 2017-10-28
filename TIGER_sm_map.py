# Thebounding box has to be set as an input-
# parameter. The bounding box of Uasin Gishu is
# set as the default coordinates
##maxlon=number 35.63
##minlon=number 34.83
##maxlat=number 0.99
##minlat=number 0.01
# Select a data between October 2014 and now, the
# script will automatically select the closest available
##S1_Tracknr=number 130
# Sentinel-1 acquisition
##DOI=string 2015-05-01
# Select the output directory
##outdir=folder

import sys

opos = sys.platform
if opos == 'win32':
    sys.path.append('%USERPROFILE%\.qgis2\processing\scripts\TIGER_SM')
elif opos == 'linux' or opos == 'linux2':
    sys.path.append('~/.qgis2/processing/scripts/TIGER_SM')

import ee
import datetime as dt
import numpy as np
import time
import pickle
import httplib2
import os
from apiclient import discovery
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage
from googleapiclient.http import MediaIoBaseDownload
import io
from qgis.core import *
from PyQt4.QtCore import *




class gdrive(object):

    def __init__(self):

        # If modifying these scopes, delete your previously saved credentials
        # at ~/.credentials/drive-python-quickstart.json
        self.SCOPES = 'https://www.googleapis.com/auth/drive'
        self.CLIENT_SECRET_FILE = 'client_secret.json'
        self.APPLICATION_NAME = 'Drive API Python Quickstart'


    def _get_credentials(self):
        """Gets valid user credentials from storage.

        If nothing has been stored, or if the stored credentials are invalid,
        the OAuth2 flow is completed to obtain the new credentials.

        Returns:
            Credentials, the obtained credential.
        """

        home_dir = os.path.expanduser('~')
        credential_dir = os.path.join(home_dir, '.credentials')
        if not os.path.exists(credential_dir):
            os.makedirs(credential_dir)
        credential_path = os.path.join(credential_dir,
                                       'drive-python-quickstart.json')

        store = Storage(credential_path)
        credentials = store.get()
        if not credentials or credentials.invalid:
            flow = client.flow_from_clientsecrets(self.CLIENT_SECRET_FILE, self.SCOPES)
            flow.user_agent = self.APPLICATION_NAME

            credentials = tools.run(flow, store)
            print('Storing credentials to ' + credential_path)
        return credentials


    def _init_connection(self):

        credentials = self._get_credentials()
        http = credentials.authorize(httplib2.Http())
        service = discovery.build('drive', 'v3', http=http)

        return(http, service)


    def print_file_list(self):

        http, service = self._init_connection()

        results = service.files().list(
            pageSize=30, fields="nextPageToken, files(id, name)").execute()
        items = results.get('files', [])
        if not items:
            print('No files found.')
        else:
            print('Files:')
            for item in items:
                print('{0} ({1})'.format(item['name'], item['id']))


    def get_id(self, filename):

        http, service = self._init_connection()

        # get list of files
        results = service.files().list(
            pageSize=50, fields="nextPageToken, files(id, name)").execute()
        items = results.get('files', [])

        # extract list of names and id and find the wanted file
        namelist = np.array([items[i]['name'] for i in range(len(items))])
        idlist = np.array([items[i]['id'] for i in range(len(items))])
        file_pos = np.where(namelist == filename)

        if len(file_pos[0]) == 0:
            return(0, filename + ' not found')
        else:
            return(1, idlist[file_pos])


    def download_file(self, filename, localpath):

        http, service = self._init_connection()

        # get file id
        success, fId = self.get_id(filename)

        if success == 0:
            print(filename + ' not found')
            return

        request = service.files().get_media(fileId=fId[0])
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print('Download %d%%.' % int(status.progress() * 100))

        fo = open(localpath, 'wb')
        fo.write(fh.getvalue())
        fo.close()


    def delete_file(self, filename):

        http, service = self._init_connection()

        # get file id
        success, fId = self.get_id(filename)

        if success == 0:
            print(filename + ' not found')

        service.files().delete(fileId=fId[0]).execute()


def extr_GEE_array_reGE(minlon, minlat, maxlon, maxlat, year, month, day, tempfilter=True, applylcmask=True,
                        sampling=20, dualpol=True, trackflt=None, maskwinter=True, masksnow=True):
    def maskterrain(image):
        # srtm dem
        gee_srtm = ee.Image("USGS/SRTMGL1_003")
        gee_srtm_slope = ee.Terrain.slope(gee_srtm)

        tmp = ee.Image(image)
        mask = gee_srtm_slope.lt(20)
        mask2 = tmp.lt(0).bitwiseAnd(tmp.gt(-25))
        # mask2 = tmp.gte(-25)
        mask = mask.bitwiseAnd(mask2)
        tmp = tmp.updateMask(mask)
        return (tmp)

    def masklc(image):
        # load land cover info
        corine = ee.Image('users/felixgreifeneder/corine')

        # create lc mask
        valLClist = [10, 11, 12, 13, 18, 19, 20, 21, 26, 27, 28, 29]

        lcmask = corine.eq(valLClist[0]).bitwiseOr(corine.eq(valLClist[1])) \
            .bitwiseOr(corine.eq(valLClist[2])) \
            .bitwiseOr(corine.eq(valLClist[3])) \
            .bitwiseOr(corine.eq(valLClist[4])) \
            .bitwiseOr(corine.eq(valLClist[5])) \
            .bitwiseOr(corine.eq(valLClist[6])) \
            .bitwiseOr(corine.eq(valLClist[7])) \
            .bitwiseOr(corine.eq(valLClist[8])) \
            .bitwiseOr(corine.eq(valLClist[9])) \
            .bitwiseOr(corine.eq(valLClist[10])) \
            .bitwiseOr(corine.eq(valLClist[11]))

        tmp = ee.Image(image)

        tmp = tmp.updateMask(lcmask)
        return (tmp)

    def setresample(image):
        image = image.resample()
        return (image)

    def toln(image):

        tmp = ee.Image(image)

        # Convert to linear
        vv = ee.Image(10).pow(tmp.select('VV').divide(10))
        if dualpol == True:
            vh = ee.Image(10).pow(tmp.select('VH').divide(10))

        # Convert to ln
        out = vv.log()
        if dualpol == True:
            out = out.addBands(vh.log())
            out = out.select(['constant', 'constant_1'], ['VV', 'VH'])
        else:
            out = out.select(['constant'], ['VV'])

        return out.set('system:time_start', tmp.get('system:time_start'))

    def tolin(image):

        tmp = ee.Image(image)

        # Covert to linear
        vv = ee.Image(10).pow(tmp.select('VV').divide(10))
        if dualpol == True:
            vh = ee.Image(10).pow(tmp.select('VH').divide(10))

        # Convert to
        if dualpol == True:
            out = vv.addBands(vh)
            out = out.select(['constant', 'constant_1'], ['VV', 'VH'])
        else:
            out = vv.select(['constant'], ['VV'])

        return out.set('system:time_start', tmp.get('system:time_start'))

    def todb(image):

        tmp = ee.Image(image)

        return ee.Image(10).multiply(tmp.log10()).set('system:time_start', tmp.get('system:time_start'))

    def applysnowmask(image):

        tmp = ee.Image(image)
        sdiff = tmp.select('VH').subtract(snowref)
        wetsnowmap = sdiff.lte(-2.6).focal_mode(100, 'square', 'meters', 3)

        return (tmp.updateMask(wetsnowmap.eq(0)))

    ee.Reset()
    ee.Initialize()

    # load S1 data
    gee_s1_collection = ee.ImageCollection('COPERNICUS/S1_GRD')

    # for lc_id in range(1, len(valLClist)):
    #     tmpmask = corine.eq(valLClist[lc_id])
    #     lcmask = lcmask.bitwiseAnd(tmpmask)

    # construct roi
    roi = ee.Geometry.Polygon([[minlon, minlat], [minlon, maxlat],
                               [maxlon, maxlat], [maxlon, minlat],
                               [minlon, minlat]])
    # roi = get_south_tyrol_roi()

    # ASCENDING acquisitions
    gee_s1_filtered = gee_s1_collection.filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .filterBounds(roi) \
        .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING')) \
        .filter(ee.Filter.eq('platform_number', 'A')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))

    if dualpol == True:
        gee_s1_filtered = gee_s1_filtered.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))

    if trackflt is not None:
        gee_s1_filtered = gee_s1_filtered.filter(ee.Filter.eq('relativeOrbitNumber_start', trackflt))

    if maskwinter == True:
        gee_s1_filtered = gee_s1_filtered.filter(ee.Filter.dayOfYear(121, 304))

    if applylcmask == True:
        gee_s1_filtered = gee_s1_filtered.map(masklc)

    gee_s1_filtered = gee_s1_filtered.map(setresample)
    gee_s1_filtered = gee_s1_filtered.map(maskterrain)

    # apply wet snow mask
    if masksnow == True:
        gee_s1_linear_vh = gee_s1_filtered.map(tolin).select('VH')
        snowref = ee.Image(10).multiply(gee_s1_linear_vh.reduce(ee.Reducer.intervalMean(5, 100)).log10())
        # snowdiff = s1_sig0_vh.subtract(snowref)
        # wetsnowmap = snowdiff.lte(-2.6).focal_mode(100, 'square', 'meters', 3)
        # s1_sig0_vv = s1_sig0_vv.updateMask(wetsnowmap.eq(0))
        # if dualpol == True:
        #    s1_sig0_vh = s1_sig0_vh.updateMask(wetsnowmap.eq(0))
        gee_s1_filtered = gee_s1_filtered.map(applysnowmask)

    # create a list of availalbel dates
    tmp = gee_s1_filtered.getInfo()
    tmp_ids = [x['properties']['system:index'] for x in tmp['features']]
    dates = np.array([dt.date(year=int(x[17:21]), month=int(x[21:23]), day=int(x[23:25])) for x in tmp_ids])

    # find the closest acquisitions
    doi = dt.date(year=year, month=month, day=day)
    doi_index = np.argmin(np.abs(dates - doi))
    date_selected = dates[doi_index]

    # filter imagecollection for respective date
    gee_s1_list = gee_s1_filtered.toList(2000)
    doi_indices = np.where(dates == date_selected)[0]
    gee_s1_drange = ee.ImageCollection(gee_s1_list.slice(np.int(doi_indices[0]), np.int(doi_indices[-1] + 1)))
    s1_sig0 = gee_s1_drange.mosaic()
    s1_sig0 = ee.Image(s1_sig0.copyProperties(gee_s1_drange.first()))

    # fetch image from image collection
    s1_lia = s1_sig0.select('angle').clip(roi)
    # get the track number
    s1_sig0_info = s1_sig0.getInfo()
    track_nr = s1_sig0_info['properties']['relativeOrbitNumber_start']

    # despeckle
    if tempfilter == True:
        radius = 7
        units = 'pixels'
        gee_s1_linear = gee_s1_filtered.map(tolin)
        gee_s1_dspckld_vv = multitemporalDespeckle(gee_s1_linear.select('VV'), radius, units,
                                                   {'before': -12, 'after': 12, 'units': 'month'})
        gee_s1_dspckld_vv = gee_s1_dspckld_vv.map(todb)
        gee_s1_list_vv = gee_s1_dspckld_vv.toList(2000)
        gee_s1_fltrd_vv = ee.ImageCollection(gee_s1_list_vv.slice(np.int(doi_indices[0]), np.int(doi_indices[-1] + 1)))
        s1_sig0_vv = gee_s1_fltrd_vv.mosaic()
        # s1_sig0_vv = ee.Image(gee_s1_list_vv.get(doi_index))

        if dualpol == True:
            gee_s1_dspckld_vh = multitemporalDespeckle(gee_s1_linear.select('VH'), radius, units,
                                                       {'before': -12, 'after': 12, 'units': 'month'})
            gee_s1_dspckld_vh = gee_s1_dspckld_vh.map(todb)
            gee_s1_list_vh = gee_s1_dspckld_vh.toList(2000)
            gee_s1_fltrd_vh = ee.ImageCollection(gee_s1_list_vh.slice(np.int(doi_indices[0]), np.int(doi_indices[-1] + 1)))
            s1_sig0_vh = gee_s1_fltrd_vh.mosaic()

        if dualpol == True:
            s1_sig0 = s1_sig0_vv.addBands(s1_sig0_vh).select(['constant', 'constant_1'], ['VV', 'VH'])
        else:
            s1_sig0 = s1_sig0_vv.select(['constant'], ['VV'])
            # s1_sig0_dsc = s1_sig0_vv_dsc.select(['constant'], ['VV'])

    # extract information
    s1_sig0_vv = s1_sig0.select('VV')
    s1_sig0_vv = s1_sig0_vv.clip(roi)
    if dualpol == True:
        s1_sig0_vh = s1_sig0.select('VH')
        s1_sig0_vh = s1_sig0_vh.clip(roi)

    # compute temporal statistics
    gee_s1_filtered = gee_s1_filtered.filterMetadata('relativeOrbitNumber_start', 'equals', track_nr)

    gee_s1_ln = gee_s1_filtered.map(toln)
    # gee_s1_ln = gee_s1_ln.clip(roi)
    k1vv = ee.Image(gee_s1_ln.select('VV').mean()).clip(roi)
    k2vv = ee.Image(gee_s1_ln.select('VV').reduce(ee.Reducer.stdDev())).clip(roi)

    if dualpol == True:
        k1vh = ee.Image(gee_s1_ln.select('VH').mean()).clip(roi)
        k2vh = ee.Image(gee_s1_ln.select('VH').reduce(ee.Reducer.stdDev())).clip(roi)

    # mask insensitive pixels
    # if dualpol == True:
    #     smask = k2vv.gt(0.4).And(k2vh.gt(0.4))
    # else:
    #     smask = k2vv.gt(0.4)
    #
    # s1_sig0_vv = s1_sig0_vv.updateMask(smask)
    # s1_lia = s1_lia.updateMask(smask)
    # k1vv = k1vv.updateMask(smask)
    # k2vv = k2vv.updateMask(smask)
    #
    # if dualpol == True:
    #     s1_sig0_vh = s1_sig0_vh.updateMask(smask)
    #     k1vh = k1vh.updateMask(smask)
    #     k2vh = k2vh.updateMask(smask)

    # export
    if dualpol == False:
        # s1_sig0_vv = s1_sig0_vv.reproject(s1_lia.projection())
        return (s1_sig0_vv, s1_lia, k1vv, k2vv, roi, date_selected)
    else:
        return (s1_lia.focal_median(sampling, 'square', 'meters'),
                s1_sig0_vv.focal_median(sampling, 'square', 'meters'),
                s1_sig0_vh.focal_median(sampling, 'square', 'meters'),
                k1vv.focal_median(sampling, 'square', 'meters'),
                k1vh.focal_median(sampling, 'square', 'meters'),
                k2vv.focal_median(sampling, 'square', 'meters'),
                k2vh.focal_median(sampling, 'square', 'meters'),
                roi,
                date_selected)


def multitemporalDespeckle(images, radius, units, opt_timeWindow=None):

    def mapMeanSpace(i):
        reducer = ee.Reducer.mean()
        kernel = ee.Kernel.square(radius, units)
        mean = i.reduceNeighborhood(reducer, kernel).rename(bandNamesMean)
        ratio = i.divide(mean).rename(bandNamesRatio)
        return (i.addBands(mean).addBands(ratio))

    if opt_timeWindow == None:
        timeWindow = dict(before=-3, after=3, units='month')
    else:
        timeWindow = opt_timeWindow

    bandNames = ee.Image(images.first()).bandNames()
    bandNamesMean = bandNames.map(lambda b: ee.String(b).cat('_mean'))
    bandNamesRatio = bandNames.map(lambda b: ee.String(b).cat('_ratio'))

    # compute spatial average for all images
    meanSpace = images.map(mapMeanSpace)

    # computes a multi-temporal despeckle function for a single image

    def multitemporalDespeckleSingle(image):
        t = image.date()
        fro = t.advance(ee.Number(timeWindow['before']), timeWindow['units'])
        to = t.advance(ee.Number(timeWindow['after']), timeWindow['units'])

        meanSpace2 = ee.ImageCollection(meanSpace).select(bandNamesRatio).filterDate(fro, to) \
            .filter(ee.Filter.eq('relativeOrbitNumber_start', image.get('relativeOrbitNumber_start')))

        b = image.select(bandNamesMean)

        return (b.multiply(meanSpace2.sum()).divide(meanSpace2.count()).rename(bandNames)).set('system:time_start',
                                                                                               image.get(
                                                                                                   'system:time_start'))

    return meanSpace.map(multitemporalDespeckleSingle)


def GEtodisk(geds, name, dir, sampling, roi):

    file_exp = ee.batch.Export.image.toDrive(image=geds, description='fileexp' + name,
                                             fileNamePrefix=name, scale=sampling, region=roi.toGeoJSON()['coordinates'],
                                             maxPixels=1000000000000)

    file_exp.start()

    start = time.time()
    success = 1

    while (file_exp.active() == True):
        time.sleep(2)
        if (time.time()-start) > 4800:
            success = 0
            break
    else:
        print('Export completed')

    if success == 1:
        # initialise Google Drive
        drive_handler = gdrive()
        print('Downloading files ...')
        print(name)
        drive_handler.download_file(name + '.tif',
                                    dir + name + '.tif')
        drive_handler.delete_file(name + '.tif')
    else:
        file_exp.cancel()


def estimateSMConline(modelpath,
                      ges1vv,
                      gek1vv,
                      gek2vv,
                      roi,
                      outpath,
                      outname,
                      sampling):

    # load SVR model
    MLmodel = pickle.load(open(modelpath, 'rb'))

    # create parameter images
    alpha = [ee.Image(MLmodel.SVRmodel.best_estimator_.dual_coef_[0][i]) for i in
             range(len(MLmodel.SVRmodel.best_estimator_.dual_coef_[0]))]
    gamma = ee.Image(-MLmodel.SVRmodel.best_estimator_.gamma)
    intercept = ee.Image(MLmodel.SVRmodel.best_estimator_.intercept_[0])

    # support vectors stack
    sup_vectors = MLmodel.SVRmodel.best_estimator_.support_vectors_
    n_vectors = sup_vectors.shape[0]

    tmp_a = ee.Image(sup_vectors[0, 0])
    tmp_b = ee.Image(sup_vectors[0, 1])
    tmp_c = ee.Image(sup_vectors[0, 2])

    sup_image = ee.Image.cat(tmp_a, tmp_b, tmp_c).select(['constant', 'constant_1', 'constant_2'],
                                                         ['VV', 'VV_1', 'VV_stdDev'])
    sup_list = [sup_image]

    for i in range(1, n_vectors):
        tmp_a = ee.Image(sup_vectors[i, 0])
        tmp_b = ee.Image(sup_vectors[i, 1])
        tmp_c = ee.Image(sup_vectors[i, 2])
        sup_image = ee.Image.cat(tmp_a, tmp_b, tmp_c).select(['constant', 'constant_1', 'constant_2'],
                                                             ['VV', 'VV_1', 'VV_stdDev'])
        sup_list.append(sup_image)

    input_image = ee.Image([ges1vv, gek1vv, gek2vv])
    ipt_img_mask = input_image.mask().reduce(ee.Reducer.allNonZero())

    S1mask = ges1vv.mask()
    zeromask = input_image.neq(ee.Image(0)).reduce(ee.Reducer.allNonZero())

    combined_mask = S1mask.And(zeromask).And(ipt_img_mask)

    input_image = input_image.updateMask(ee.Image([combined_mask, combined_mask,
                                                   combined_mask]))

    # scale the estimation image
    scaling_std_img = ee.Image([ee.Image(MLmodel.scaler.scale_[0].astype(np.float)),
                                ee.Image(MLmodel.scaler.scale_[1].astype(np.float)),
                                ee.Image(MLmodel.scaler.scale_[2].astype(np.float))])

    scaling_std_img = scaling_std_img.select(['constant', 'constant_1', 'constant_2'],
                                             ['VV', 'VV_1', 'VV_stdDev'])

    scaling_mean_img = ee.Image([ee.Image(MLmodel.scaler.mean_[0].astype(np.float)),
                                 ee.Image(MLmodel.scaler.mean_[1].astype(np.float)),
                                 ee.Image(MLmodel.scaler.mean_[2].astype(np.float))])

    scaling_mean_img = scaling_mean_img.select(['constant', 'constant_1', 'constant_2'],
                                               ['VV', 'VV_1', 'VV_stdDev'])

    input_image_scaled = input_image.subtract(scaling_mean_img).divide(scaling_std_img)

    k_x1x2 = [sup_list[i].subtract(input_image_scaled) \
                  .pow(ee.Image(2)) \
                  .reduce(ee.Reducer.sum()) \
                  .sqrt() \
                  .pow(ee.Image(2)) \
                  .multiply(ee.Image(gamma)) \
                  .exp() for i in range(n_vectors)]

    alpha_times_k = [ee.Image(alpha[i].multiply(k_x1x2[i])) for i in range(n_vectors)]

    alpha_times_k = ee.ImageCollection(alpha_times_k)
    alpha_times_k_sum = alpha_times_k.reduce(ee.Reducer.sum())

    estimated_smc = alpha_times_k_sum.add(intercept).round().int8()

    GEtodisk(estimated_smc, outname, outpath, sampling, roi)



DDOI = dt.datetime.strptime(DOI, '%Y-%m-%d')
year = DDOI.year
month = DDOI.month
day = DDOI.day

sampling = 100
tracknr = S1_Tracknr

outpath = outdir +'/'

images = extr_GEE_array_reGE(minlon, minlat, maxlon, maxlat,
                                         year, month, day,
                                         tempfilter=True,
                                         applylcmask=False,
                                         sampling=sampling,
                                         dualpol=False,
                                         trackflt=tracknr,
                                         maskwinter=False,
                                         masksnow=False)

outname = 'SMCmap_' +str(int(minlon)) + str(int(minlat)) + str(int(maxlon)) + str(int(maxlat)) + '_' \
          + str(images[5].year) + '_' + str(images[5].month) + '_' + str(images[5].day)

progress.setInfo('Starting to generate map')

estimateSMConline("C:/Users/FGreifeneder/.qgis2/processing/scripts/TIGER_SM/SVR_Model_Python_S1.p",
                          images[0],
                          images[2],
                          images[3],
                          images[4],
                          outpath,
                          outname,
                          sampling)

progress.setInfo( outpath + outname + '.tif')

#outmap = processing.getObject( outpath + outname + '.tif' )
outmap = outpath + outname + '.tif'

##outmap=output raster
