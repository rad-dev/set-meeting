
from contextvars import ContextVar
import datetime
from io import BytesIO
import json
import math
import os
from typing import Any

import cv2
import numpy as np
from parse import parse
from polycircles import polycircles
import requests
from scipy.spatial import Voronoi
import simplekml
from sklearn.cluster import DBSCAN
import sseclient
from tqdm import tqdm

# Disable flag warning
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

from .utility import *

from engine.progress_observer import ProgressManager
from engine.source.map_handler.map_base import MapBase


class ES_Correction:
    """
    """

    def __init__(self, api_url, api_key, proxies):
        """
        Initializer
        """
        self.BASE_DIR = os.path.dirname(__file__)
        self.image_dir = os.path.join(self.BASE_DIR, "satellite_images")
        make_dir(self.image_dir)

        self.address_db_path = "Addres_DB.csv"

        self.IMAGE_SIZE = 640
        self.TILESIZE = float(256)  # Constant from google map
        self.API_URL = api_url
        self.API_KEY = api_key
        self.proxies = proxies
        self.progress_id = ContextVar('progress_id', default=None)

    def es_location_detection(self,
                              es_list,
                              dupl_dist_thrsh,
                              exp_thrsh,
                              z_level,
                              map_fetcher=MapBase,
                              api_list=["Google"],
                              stride=0.9,
                              full_range_search=False,
                              progress_monitor:Any=object):
        """
        Main function

        :param es_list: given ES antenna gps coordinate list
        :type es_list: list (tuple (lat, lon))
        :param dupl_dist_thrsh: duplicate check distance threshold (meter)
        :type dupl_dist_thrsh: float
        :param exp_thrsh: search range expansion threshold
        :type exp_thrsh: int
        :param z_level: zoom level of satellite image
        :type z_level: int
        :param stride: ratio for search step (1.0 == IMAGE WIDTH)
        :type stride: float
        :param full_range_search:
        """

        detected_latlon_list = []
        detected_bbox_list = []

        map_fetcher_call = map_fetcher(self.transform_pixel_to_gps,
                                       self.get_antenna_poses_in_img,
                                       api_list, z_level, self.IMAGE_SIZE,
                                       progress_monitor=progress_monitor)

        current_progress = 5
        progress_monitor.progress("Preparing detection functions and utilities", current_progress)
        stride_pix = int(self.IMAGE_SIZE * stride)

        final_detection_progress_percent = 78
        progress_increment_per_es = final_detection_progress_percent / len(es_list)

        # progress_monitor.progress("Fetching and processing images for earth stations", 10)
        for idx, es in enumerate(tqdm(es_list)):
            current_progress += progress_increment_per_es * 0.25
            detected_list_from_one = {}
            es["image_list"] = set()
            progress_monitor.progress(f"Fetching and processing images for earth station {idx + 1} of {len(es_list)}",
                                      current_progress)
            for exp_lv in range(0, exp_thrsh + 1):
                lat, lon = es["LatLon"]
                set_detected_list, image_list = map_fetcher_call.get_and_save_satellite_image_list(lat,
                                                                                                   lon,
                                                                                                   exp_lv,
                                                                                                   stride_pix)
                es["image_list"].update(image_list)
                detected_list_from_one.update(set_detected_list)

                if not full_range_search and len(detected_list_from_one) > 0:
                    break

            current_progress += progress_increment_per_es * 0.65
            progress_monitor.progress(f"Removing duplicate earth stations", current_progress)
            for detected_latlon, detected_bbox in detected_list_from_one.items():
                if not self.duplicate_check(detected_latlon, detected_latlon_list, dupl_dist_thrsh):
                    detected_latlon_list.append(detected_latlon)
                    detected_bbox_list.append(detected_bbox)
            es["image_list"] = list(es["image_list"])

            current_progress += progress_increment_per_es * 0.1

        dt_list = []
        progress_monitor.progress(f"Updating earth station objects", 85)
        for detected_latlon, detected_bbox in zip(detected_latlon_list, detected_bbox_list):
            dt_list.append({'Lat': detected_latlon[0],
                            'Lon': detected_latlon[1],
                            'LatLon': detected_latlon,
                            'Bbox': detected_bbox['bounding_box'],
                            'ImageRef': detected_bbox['image_name']})

        self.calibration_using_semantic_map(dt_list)

        return dt_list, es_list

    def get_img_from_GoogleMapAPI(self, gps_coord, z_level):
        """
        Get Image using GPS coordinate and zoom level.
        :param gps_coord: gps coordinate
        :type gps_coord: tuple (lat, lon)
        :param z_level: zoom level of satellite image
        :type z_level: int
        :param api_key: Google Maps Static API key
        :type api_key: str
        :return: satellite image
        :rtype: cv2.Image
        """
        _img = None

        # Google Maps Static API longitudes and latitudes have a precision of up to 6 decimal
        # places only. (https://developers.google.com/maps/documentation/maps-static/start#Locations)
        _lat, _lon = [round(x, 6) for x in gps_coord]

        _URL = "https://maps.googleapis.com/maps/api/staticmap?" + \
               f"center={_lat},{_lon}&" + \
               "scale=1&" + \
               f"zoom={z_level}&" + \
               f"size={self.IMAGE_SIZE}x{self.IMAGE_SIZE}&" + \
               "format=jpg-baseline&" + \
               "maptype=satellite&" + \
               f"key={self.API_KEY}"
        
        print(_URL)
        _response = requests.get(_URL, proxies=self.proxies, verify=False)

        if _response.status_code == 200:
            if int(_response.headers.get('Content-Length')) == len(BytesIO(_response.content).getvalue()):
                _np_img = np.frombuffer(BytesIO(_response.content).read(), dtype=np.uint8)
                _img = cv2.imdecode(_np_img, cv2.IMREAD_COLOR)
            else:
                print("[es_correct.py ERROR] Content length does not match satellite image file size")
        else:
            print("[es_correct.py ERROR] Cannot fetch satellite image")

        return _img

    def get_antenna_poses_in_img(self,
                                 img,
                                 img_filename="sat_img.jpg",
                                 api_url=None,
                                 progress_monitor=object):
        """
        Get detected position of antennas using deep-learning model
        Set the left-top as the reference point.
        right value is x, lower value is y
        :param img: satellite image
        :type img: cv2.Image
        :param img_filename: filename of image when saved to the server
        :type img_filename: str
        :return ct_bboxes: list of [x, y] (x, y: 0~IMAGE_SIZE-1), so center of image would be [IMAGE_SIZE/2, IMAGE_SIZE/2]
                           if no antenna, return []
        :rtype ct_bboxes: list (list ([, ]))
        """

        ct_bboxes = []
        sat_bboxes = []
        int_sat_bboxes = []

        if api_url is not None:

            # Server is currently hosted at AOA
            UPLOAD_API = f"{api_url}/api/esd/upload/single/"
            DETECT_API = f"{api_url}/api/esd/detect/"
            PROGRESS_API = f"{api_url}/api/esd/status"

            _, _jpg_image = cv2.imencode('.jpg', img)
            _send_image = {"file": (img_filename, _jpg_image.tostring(), "image/jpeg")}
            _response_upload = requests.post(UPLOAD_API, files=_send_image)
            print(UPLOAD_API,_response_upload.status_code)

            if _response_upload.status_code == 200:
                current_datetime = datetime.datetime.now()
                message_id = current_datetime.strftime("%m%d%Y%H%M%S")
                progress_client = sseclient.SSEClient(f"{PROGRESS_API}/{message_id}")
                event_progress = 0

                _response_detect = None
                while event_progress != 100:
                    try:
                        event = next(progress_client)
                        resp = json.loads(event.data.replace("\'", "\""))
                        event_progress = resp["progress"]

                        if event_progress == 0:
                            _file_list = {
                                "image_list": [img_filename],
                                "progress_id": message_id
                            }
                            _response_detect = requests.post(DETECT_API, json=_file_list)
                        elif event_progress == 100:
                            if _response_detect is not None:
                                _detect = _response_detect.json()
                                _im_height, _im_width, _ = img.shape
                                sat_bboxes = [[y["bbox"][0] * _im_width, y["bbox"][1] * _im_height,
                                            y["bbox"][2] * _im_width, y["bbox"][3] * _im_height]
                                            for x in _detect for y in x["bboxes"]]
                                ct_bboxes = [[int((x[0] + x[2]) / 2), int((x[1] + x[3]) / 2)] for x in sat_bboxes]
                                int_sat_bboxes = [[[int(x[0]), int(x[1])], [int(x[2]), int(x[3])]] for x in sat_bboxes]
                            else:
                                print("[es_correct.py ERROR] Cannot detect satellites")
                            
                            progress_client.resp.close()

                    except Exception as e:
                        progress_client.resp.close()
                        raise e
            else:
                print("[es_correct.py ERROR] Cannot upload satellite image to server")
        else:
            print("[es_correct.py ERROR] No specified API URL")

        return ct_bboxes, int_sat_bboxes

    def transform_pixel_to_gps(self, center_gps, pixel_pos, z_level):
        """
        Transform pixel movement to GPS coordinate movement.
        :param pixel_pos: distance of moving pixels, top-left reference
        :type pixel_pos: list([x, y])
        :param center_gps: gps coordinate of center point
        :type center_gps: tuple(lat, lon)
        :param z_level: zoom level of satellite image
        :type z_level: int
        :return: moved gps coordinate
        :rtype: tuple(lat, lon)
        """

        sin_y = min(max(math.sin(math.radians(center_gps[0])), -0.9999), 0.9999)
        cnt_X_tot = self.TILESIZE * (0.5 + center_gps[1] / 360.0) * (1 << z_level)
        cnt_Y_tot = self.TILESIZE * (0.5 - math.log((1.0 + sin_y) / (1.0 - sin_y)) / (4.0 * math.pi)) * (1 << z_level)
        x_w = (cnt_X_tot + (pixel_pos[0] - self.IMAGE_SIZE / 2)) / float(1 << z_level)
        y_w = (cnt_Y_tot + (pixel_pos[1] - self.IMAGE_SIZE / 2)) / float(1 << z_level)
        k = math.exp(- (y_w / self.TILESIZE - 0.5) * (4.0 * math.pi))
        _lon = (x_w / self.TILESIZE - 0.5) * 360
        _lat = math.degrees(math.asin((k - 1) / (k + 1)))

        return tuple([_lat, _lon])

    def get_distance_between_gps(self, gps_coord1, gps_coord2):
        """
        Calculate distance between two GPS coordinates using Haversine Formula.
        :param gps_coord1: 1st gps coordinate
        :type gps_coord2: tuple (lat, lon)
        :param gps_coord2: 2nd gps coordinate
        :type gps_coord2: tuple (lat, lon)
        :return: float: distance between two gps coordinates
        """

        earth_radius = 6371 * 1000

        lat1, lon1 = np.deg2rad(gps_coord1[0]), np.deg2rad(gps_coord1[1])
        lat2, lon2 = np.deg2rad(gps_coord2[0]), np.deg2rad(gps_coord2[1])

        delta_lon = lon2 - lon1
        delta_lat = lat2 - lat1

        sin_delta = np.sin(delta_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
        s_root = 2 * np.arctan2(np.sqrt(sin_delta), np.sqrt(1 - sin_delta))

        return earth_radius * s_root

    def duplicate_check(self, node, check_list, dist_thrsh):
        """
        Check GPS coordinate is in List.
        :param node: new antenna position
        :type node: tuple (lat, lon)
        :param check_list: retained list of antenna gps coordinates
        :type check_list: list (tuple (lat, lon))
        :param dist_thrsh: distance criteria to determine the antenna in the same position (meter)
        :type dist_thrsh: float
        :return: if it is duplicated False otherwise True.
        :rtype: bool
        """

        for check in check_list:
            if self.get_distance_between_gps(node, check) < dist_thrsh:
                return True

        return False

    def search_neighbor(self, ori_gps, exp_lv, stride_pix, z_level):
        search_range_list = []
        for e_step in range(-exp_lv, exp_lv + 1):
            for n_step in range(-exp_lv, exp_lv + 1):
                if np.abs(e_step) == exp_lv or np.abs(n_step) == exp_lv:  # add only outside ring
                    trans_gps = self.transform_pixel_to_gps(ori_gps,
                                                            [e_step * stride_pix + (self.IMAGE_SIZE >> 1),
                                                             n_step * stride_pix + (self.IMAGE_SIZE >> 1)],
                                                            z_level)
                    search_range_list.append(trans_gps)

        return search_range_list

    def fcc_preprocessing(self,
                          fcc_data,
                          null_gps_exclude=True,
                          discontinue_es_exclude=False,
                          gps_base_clustering=True,
                          progress_monitor:Any=object):
        """
        Eliminate useless data of FCC List and Clustering the list according to gps position.
        :param fcc_data:
        :param null_gps_exclude: if data has 0 latlon or null latlon, eliminate the data
        :param discontinue_es_exclude: if data has "Discontinue C Band" attribute, eliminate the data
        :param gps_base_clustering:
        :return: pre-processed fcc data
        """

        progress_monitor.progress("Removing unneeded coordinates", 5)
        if null_gps_exclude or discontinue_es_exclude:
            for _d in fcc_data:
                if _d['Lat'] == 0 and null_gps_exclude:
                    fcc_data.remove(_d)
                    continue
                if _d["Intended Action"] == "Discontinue C Band" and discontinue_es_exclude:
                    fcc_data.remove(_d)

        if gps_base_clustering:
            progress_monitor.progress("Clustering potential earth station areas", 10)
            fcc_clustering = DBSCAN(eps=0.001, min_samples=1).fit([node['LatLon'] for node in fcc_data])
            num_label = max(fcc_clustering.labels_) + 1

            cluster_fcc_list = [{'Index': i,
                                'LatLon': (0, 0),
                                'Lat': 0,
                                'Lon': 0,
                                'Quantity': 0,
                                'Matching_DT_Index': [],
                                'ApplicantName': "",
                                'mLatLon': [],
                                'mApplicantName': [],
                                'mIndex': []}
                                for i in range(num_label)]

            # num_of_label = [list(input_clustering.labels_).count(_label) for _label in range(num_label)]

            progress_monitor.progress("Writing cluster details", 50)
            for _idx, cluster_label in enumerate(fcc_clustering.labels_):
                cluster_fcc_list[cluster_label]['Quantity'] += fcc_data[_idx]['Quantity']
                cluster_fcc_list[cluster_label]['mLatLon'].append(fcc_data[_idx]['LatLon'])
                cluster_fcc_list[cluster_label]['mApplicantName'].append(fcc_data[_idx]['ApplicantName'])
                cluster_fcc_list[cluster_label]['mIndex'].append(fcc_data[_idx]['Index'])

            progress_monitor.progress("Updating cluster coordinate details", 80)
            for cf in cluster_fcc_list:
                app_name_cntr = Counter(cf['mApplicantName'])
                max_cnt = 0
                for app_name in app_name_cntr.keys():
                    if app_name_cntr[app_name] > max_cnt:
                        max_cnt = app_name_cntr[app_name]
                        cf['ApplicantName'] = app_name

                median_latlon = tuple()
                min_diff = 10000000
                for candi_latlon in cf['mLatLon']:
                    tmp_diff = 0
                    for othr_latlon in cf['mLatLon']:
                        tmp_diff += self.get_distance_between_gps(candi_latlon, othr_latlon)
                    if tmp_diff < min_diff:
                        median_latlon = candi_latlon
                        min_diff = tmp_diff

                cf['LatLon'] = median_latlon
                cf['Lat'] = median_latlon[0]
                cf['Lon'] = median_latlon[1]

            fcc_data = cluster_fcc_list
            
            progress_monitor.progress("Done", 100)

        return fcc_data

    def es_matching(self,
                    input_data,
                    detected_data,
                    matching_thrsh=100,
                    bigger_thrsh=500,
                    grading_flag=False,
                    progress_monitor:Any=object):
        """
        Matching input data and detected ES data
        :param input_data:
        :param detected_data:
        :param matching_thrsh: threshold value for matching (meter)
        :param bigger_thrsh: threshold value for larger circle (meter)
        :param grading_flag: turn on ES grading
        :return:
        """

        get_distances_overall_percentage = 50
        current_percentage = 0
        get_distances_increment_percentage = (get_distances_overall_percentage - current_percentage) / len(detected_data)
        for dt_idx, detected in enumerate(detected_data):
            current_percentage += get_distances_increment_percentage * 0.5
            progress_monitor.progress(f"Calculating distance between cluster and earth station {dt_idx} of {len(detected_data)}",
                                      current_percentage)
            min_dist = 100000000
            matching_input = dict()
            for _input in input_data:
                tmp_dist = self.get_distance_between_gps(detected['LatLon'], _input['LatLon'])
                if tmp_dist < min_dist:
                    min_dist = tmp_dist
                    matching_input = _input

                    if matching_input.get('Outer_Matching_DT_Index', None) == None:
                        matching_input['Outer_Matching_DT_Index'] = []

            detected['Index'] = dt_idx
            
            current_percentage += get_distances_increment_percentage * 0.5
            progress_monitor.progress(f"Getting the relative distance of earth station {dt_idx} of {len(detected_data)}",
                                      current_percentage)
            if min_dist <= matching_thrsh:
                rel_dist = self.get_voronoi_distance(matching_input['LatLon'], detected['LatLon'])
                detected['Matching_FCC_Index'] = matching_input['Index']
                detected['FCC_Lat'] = matching_input['Lat']
                detected['FCC_Lon'] = matching_input['Lon']
                detected['FCC_Quantity'] = matching_input['Quantity']
                detected['ApplicantName'] = matching_input['ApplicantName']
                detected['Rel_Dist'] = rel_dist
                detected['Dist'] = min_dist
                matching_input['Matching_DT_Index'].append(dt_idx)
            elif min_dist <= bigger_thrsh:
                rel_dist = self.get_voronoi_distance(matching_input['LatLon'], detected['LatLon'])
                detected['Matching_FCC_Index'] = matching_input['Index']
                detected['FCC_Lat'] = matching_input['Lat']
                detected['FCC_Lon'] = matching_input['Lon']
                detected['FCC_Quantity'] = matching_input['Quantity']
                detected['ApplicantName'] = matching_input['ApplicantName']
                detected['Rel_Dist'] = rel_dist
                detected['Dist'] = min_dist
                matching_input['Outer_Matching_DT_Index'].append(dt_idx)
            else:
                detected['Matching_FCC_Index'] = []
                detected['FCC_Lat'] = ""
                detected['FCC_Lon'] = ""
                detected['FCC_Quantity'] = ""
                detected['ApplicantName'] = ""
                detected['Rel_Dist'] = ""
                detected['Dist'] = ""

        matching_overall_percentage = 95
        get_distances_increment_percentage = (matching_overall_percentage - current_percentage) / len(input_data)
        for _input_idx, _input in enumerate(input_data):
            current_percentage += get_distances_increment_percentage * 0.85
            progress_monitor.progress(f"Matching earth stations to cluster {_input_idx} of {len(input_data)}",
                                      current_percentage)
            _input['Number of DT'] = len(_input['Matching_DT_Index'])
            _input['Outer Number of DT'] = len(_input.get('Outer_Matching_DT_Index', []))

            if _input['Number of DT'] != 0:
                dt_per_ip = [i['LatLon'] for i in detected_data if i['Index'] in _input['Matching_DT_Index']]
                site_clustering = DBSCAN(eps=0.001, min_samples=1).fit(dt_per_ip)
                num_site = max(site_clustering.labels_) + 1
            else:
                num_site = 0

            if _input['Outer Number of DT'] != 0:
                outer_dt_per_ip = [i['LatLon'] for i in detected_data if i['Index'] in _input['Outer_Matching_DT_Index']]
                outer_site_clustering = DBSCAN(eps=0.001, min_samples=1).fit(outer_dt_per_ip)
                outer_num_site = max(outer_site_clustering.labels_) + 1
            else:
                outer_num_site = 0

            if grading_flag:
                current_percentage += get_distances_increment_percentage * 0.15
                progress_monitor.progress(f"Grading cluster {_input_idx} of {len(input_data)}",
                                          current_percentage)
                if num_site != 0:
                    if num_site == 1:
                        quantity = _input['Number of DT'] if outer_num_site == 1 else _input['Number of DT'] + _input['Outer Number of DT']
                        _input['Grade'] = 'A' if quantity == _input['Quantity'] else 'B'
                    else:
                        _input['Grade'] = 'C'
                else:
                    _input['Grade'] = 'D' if outer_num_site > 0 else 'F'

            _input['Matching_DT_Index'] += _input.pop('Outer_Matching_DT_Index', [])
            _input['Number of DT'] += _input.pop('Outer Number of DT')

            for dt_idx in _input['Matching_DT_Index']:
                detected_data[dt_idx]['Grade'] = _input['Grade'] if grading_flag else ''

        self.log_writer(input_data, detected_data)

    def save_satellite_image_in_repo(self, tar_gps, z_level):
        img_name = self.make_satellite_imgname(tar_gps, z_level)
        img_path = os.path.join(self.image_dir, img_name)
        
        # check image is in repo
        if os.path.isfile(img_path):
            satellite_img = cv2.imread(img_path)
        else:
            # get image using Google_API.
            satellite_img = self.get_img_from_GoogleMapAPI(tar_gps, z_level)
            if satellite_img is not None:
                cv2.imwrite(img_path, satellite_img)

        return img_name, satellite_img

    def calibration_using_semantic_map(self, es_list):
        """
        TO BE DEVELOPED
        Calibrate position and get altitude using Semantic Map
        :param gps_list:
        :type gps_list: list (tuple (lat, lon))
        :return:
        :rtype: list (tuple (lat, lon))
        """
        for es in es_list:
            es['Altitude'] = 0

        return es_list

    def make_satellite_imgname(self, gps_coord, z_level):
        """
        Name satellite image regarding GPS coordinate and zoom level.
        :param gps_coord: gps coordinate of center of satellite image
        :type gps_coord: tuple (lat, lon)
        :param z_level: zoom level of satellite image
        :type z_level: int
        :return: image name
        :rtype: str
        """
        return "SAT_%0.8f_%0.8f_z%d.jpg" % (gps_coord[0], gps_coord[1], z_level)

    def make_voronoi_diagram(self, input_data):
        vor_gps_list = list(set([i['LatLon'] for i in input_data]))

        print("- Boronoi Diagram")
        print(f"Number of Input List: {len(input_data)}")
        self.vor = Voronoi(vor_gps_list)
        self.make_voronoi_kml()
        print("Done")

    def get_voronoi_distance(self, base_gps, detected_gps):
        if base_gps == detected_gps:
            return 0

        # find base region
        _idx = 0
        for i, vp in enumerate(self.vor.points):
            if is_equal_gps(vp, base_gps):
                _idx = i
                break

        base_region = self.vor.regions[self.vor.point_region[_idx]]
        boundaries = self.vor.vertices[base_region]

        inter_pt = None
        line_b2d = [base_gps, detected_gps]

        # boundary outside
        for i in range(len(boundaries)):
            line_b2b = [boundaries[i], boundaries[(i + 1) % len(boundaries)]]
            it_pt = find_intersections(line_b2d, line_b2b)
            if it_pt:
                inter_pt = it_pt

        # boundary inside
        if inter_pt is None:
            vec_ = np.array(detected_gps) - np.array(base_gps)
            ext_detected_gps = np.array(detected_gps) + (vec_ * 10000000.0)
            line_b2d = [base_gps, ext_detected_gps]

            for i in range(len(boundaries)):
                line_b2b = [boundaries[i], boundaries[(i + 1) % len(boundaries)]]
                it_pt = find_intersections(line_b2d, line_b2b)
                if it_pt:
                    inter_pt = it_pt

        if inter_pt is None:
            return -1.0
        else:
            dist_b2d = self.get_distance_between_gps(base_gps, detected_gps)
            dist_b2b = self.get_distance_between_gps(base_gps, inter_pt)

            ratio_ = dist_b2d / dist_b2b

            return ratio_

    def make_voronoi_kml(self):
        kml = simplekml.Kml()

        vor_region = kml.newmultigeometry(name="Voronoi_Region")
        for v_r in self.vor.regions:
            if len(v_r) >= 2:
                out_bd = [(self.vor.vertices[_v][1], self.vor.vertices[_v][0]) for _v in v_r] + \
                         [(self.vor.vertices[v_r[0]][1], self.vor.vertices[v_r[0]][0])]
                vor_region.newpolygon(outerboundaryis=out_bd)
        vor_region.style.linestyle.color = simplekml.Color.yellow
        vor_region.style.linestyle.width = 3
        vor_region.style.polystyle.color = simplekml.Color.changealphaint(50, simplekml.Color.yellow)

        vor_points = kml.newmultigeometry(name="Voronoi_Points")
        vor_points.style.labelstyle.scale = 0  # Remove the labels from all the points
        vor_points.style.iconstyle.color = simplekml.Color.yellow
        vor_points.style.iconstyle.scale = 0.5
        for v_r in self.vor.points:
            vor_points.newpoint(coords=[(v_r[1], v_r[0])])

        kml.save("Voronoi.kml")

    def log_writer(self, ip_list, dt_list):
        """
        Write log of test
        :param ip_list:
        :param dt_list:
        :return:
        """
        log_dir = os.path.join(self.BASE_DIR, "log")
        make_dir(log_dir)

        _now = datetime.datetime.now()
        _now = _now.strftime('%Y%m%d%H%M')

        log_dir = os.path.join(log_dir, "log_" + _now)
        make_dir(log_dir)

        fcc_dir = os.path.join(log_dir, f"fcc_{_now}.csv")
        dt_dir = os.path.join(log_dir, f"dt_{_now}.csv")
        kml_dir = os.path.join(log_dir, f"kml_{_now}.kml")

        result_dir = os.path.join(log_dir, f"SAMSUNG ES PROTECTION SOLUTION_{_now}.csv")

        self.save_csv_kml(ip_list, dt_list, fcc_dir, dt_dir, kml_dir, result_dir)


    def save_csv_kml(self, input_data, detected_data, fcc_dir, dt_dir, kml_dir, res_dir, search_radius=100):
        """
        Save csv file and kml file of input data and detected data
        :param input_data:
        :param detected_data:
        :param fcc_dir:
        :param dt_dir:
        :param kml_dir:
        :param res_dir:
        :return:
        """
        kml = simplekml.Kml()

        fcc_points = kml.newfolder(name="FCC_Cluster_Median_Points")
        for _input in input_data:
            fcc_pt = fcc_points.newpoint(name=_input['Grade'],
                                         coords=[(_input['Lon'], _input['Lat'])],
                                         description=f"idx: {_input['Index']}\n"
                                                     f"lat: {_input['Lat']:.8f}\n"
                                                     f"lon: {_input['Lon']:.8f}\n"
                                                     f"Q:{_input['Quantity']}|D:{_input['Number of DT']}")
            fcc_pt.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png'
            fcc_pt.style.labelstyle.scale = 1
            if _input['Quantity'] == _input['Number of DT']:
                fcc_pt.style.iconstyle.color = simplekml.Color.white
            elif _input['Quantity'] < _input['Number of DT']:
                fcc_pt.style.iconstyle.color = simplekml.Color.blue
            elif _input['Number of DT'] == 0:
                fcc_pt.style.iconstyle.color = simplekml.Color.red
            else:
                fcc_pt.style.iconstyle.color = simplekml.Color.yellow

            fcc_pt.style.iconstyle.scale = 1

        dt_points = kml.newfolder(name="Detected_Points")
        for detected in detected_data:
            try:
                dt_pt = dt_points.newpoint(coords=[(detected['Lon'], detected['Lat'])],
                                           description=f"Index: {detected['Index']}\n"
                                                       f"EucDist: {detected['Dist']:.1f}m\n"
                                                       f"RelDist: {detected['Rel_Dist']:.4f}")
            except:
                dt_pt = dt_points.newpoint(name=f"No Match", coords=[(detected['Lon'], detected['Lat'])])
                pass
            dt_pt.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/placemark_square.png'
            dt_pt.style.labelstyle.scale = 0.5
            dt_pt.style.iconstyle.color = simplekml.Color.limegreen
            dt_pt.style.iconstyle.scale = 1

        match_lines = kml.newfolder(name="Match_Lines")
        for _input in input_data:
            for m_idx in _input['Matching_DT_Index']:
                match_dt = next((d_d for d_d in detected_data if d_d['Index'] == m_idx), False)
                try:
                    coords = [(match_dt['Lon'], match_dt['Lat']), (_input['Lon'], _input['Lat'])]

                    ml = match_lines.newlinestring(coords=coords)
                    ml.extrude = 1
                    ml.style.linestyle.width = 2
                    ml.style.linestyle.color = simplekml.Color.limegreen
                except TypeError:
                    continue

        circles = kml.newfolder(name=f"{search_radius}m_Boundary")
        for _input in input_data:
            polycircle = polycircles.Polycircle(latitude=_input['Lat'],
                                                longitude=_input['Lon'],
                                                radius=search_radius,
                                                number_of_vertices=36)
            p_circle = circles.newpolygon(outerboundaryis=polycircle.to_kml())
            p_circle.style.polystyle.color = simplekml.Color.changealphaint(64, simplekml.Color.green)

        kml.save(kml_dir)

        if len(detected_data) != 0:
            dict_list_to_csv(dt_dir, detected_data)
        if len(input_data) != 0:
            dict_list_to_csv(fcc_dir, input_data)
            dict_list_to_std_form(res_dir, input_data, detected_data)

    def read_address_db(self):
        self.address_db_dict = {}
        if os.path.isfile(self.address_db_path):
            with open(self.address_db_path) as csv_content:
                input_data = csv.DictReader(csv_content)
                for line in input_data:
                    self.address_db_dict[line['Address']] = (float(line['Lat']), float(line['Lon']))

    def get_address_gps_from_GoogleMapAPI(self, api_key, address, postcode, state):
        add_gps = (0, 0)

        _URL = "https://maps.googleapis.com/maps/api/geocode/json?" + \
               f"address={address}&" + \
               "components=" \
               f"postal_code={postcode}|" + \
               f"administrative_area_level_1={state}" \
               f"&key={api_key}"

        _response = requests.get(_URL, proxies=self.proxies, verify=False)

        if _response.status_code == 200:
            _msg = json.loads(_response.text)
            try:
                add_gps = _msg['results'][0]['geometry']['location']
                add_gps = (add_gps['lat'], add_gps['lng'])
            except:
                pass
        else:
            print("[es_correct.py ERROR] Cannot get GPS coordinates")
            add_gps = (0, 0)

        return add_gps

    def write_address_db(self):
        with open(self.address_db_path, "w", newline='') as f:
            wr = csv.writer(f)
            wr.writerow(["Address", "Lat", "Lon"])
            for _add in self.address_db_dict.keys():
                wr.writerow([_add, self.address_db_dict[_add][0], self.address_db_dict[_add][1]])

    def make_address_kml(self, log_dir):
        kml = simplekml.Kml()

        add_points = kml.newmultigeometry(name="Address_Points")
        add_points.style.labelstyle.scale = 0  # Remove the labels from all the points
        add_points.style.iconstyle.color = simplekml.Color.green
        add_points.style.iconstyle.scale = 0.5

        line_address = kml.newmultigeometry(name="Address_Lines")
        line_address.extrude = 1
        line_address.altitudemode = simplekml.AltitudeMode.relativetoground
        line_address.style.linestyle.width = 2
        line_address.style.linestyle.color = simplekml.Color.green

        for _key in self.kml_add_dict.keys():
            ori_gps, add_gps = self.kml_add_dict[_key]
            add_points.newpoint(coords=[(add_gps[1], add_gps[0])])

            coords = [(add_gps[1], add_gps[0]),
                      (ori_gps[1], ori_gps[0])]
            line_address.newlinestring(coords=coords)

        kml.save(os.path.join(log_dir, "Address.kml"))

    def address_recoder(self, es_list):
        self.read_address_db()

        for es in es_list:
            add_key = f"{es['Address']},{es['Site Zip Code']},{es['Site State']}"
            try:
                gps_from_add = self.address_db_dict[add_key]
            except:
                gps_from_add = self.get_address_gps_from_GoogleMapAPI(api_key=self.API_KEY,
                                                                      address=es['Address'],
                                                                      postcode=es['Site Zip Code'],
                                                                      state=es['Site State'])
                self.address_db_dict[add_key] = gps_from_add

            es['Address_LatLon'] = gps_from_add

        self.write_address_db()

    def make_gt_list_from_labeled_img(self, data_path, dist_thrsh=2.0):
        """
        Make ground truth using satellite and corresponding json(made from LabelMe)
        :param data_path: folder path
        :param dist_thrsh:
        :return: ground truth gps list
        :rtype list (tuple (lat, lon))
        """
        gt_list = []
        img_name_list = [i for i in os.listdir(data_path) if '.jpg' in i]

        for img_name in img_name_list:
            json_name = os.path.splitext(img_name)[0] + ".json"
            json_path = os.path.join(data_path, json_name)
            if os.path.isfile(json_path):
                gps_coord, z_level = self.get_property_from_imgname(img_name)

                with open(json_path, "r") as json_file:
                    pose_list = json.load(json_file)['shapes']

                for pose in pose_list:
                    pix = pose['points']
                    pix_x = int((pix[0][0] + pix[1][0]) / 2)
                    pix_y = int((pix[0][1] + pix[1][1]) / 2)
                    gt_gps = self.transform_pixel_to_gps(gps_coord, [pix_x, pix_y], z_level)

                    if not self.duplicate_check(gt_gps, gt_list, dist_thrsh):
                        gt_list.append(gt_gps)

        return gt_list

    def get_property_from_imgname(self, img_name):
        """
        Parse GPS coordinate and zoom level from image name.
        :param img_name: image name
        :type img_name: str
        :return: gps coordinate, zoom level
        :rtype: tuple (lat, lon), int
        """
        if "bing" in img_name:
            parse_res = parse("bing_SAT_{}_{}_z{}.jpg", img_name)
        else:
            parse_res = parse("SAT_{}_{}_z{}.jpg", img_name)

        gps_coord = (float(parse_res[0]), float(parse_res[1]))
        z_level = int(parse_res[2])

        return gps_coord, z_level

    def evaluator(self, result_gps_list, gt_gps_list, dist_thrsh):
        tp = 0
        for gt in gt_gps_list:
            if self.duplicate_check(gt, result_gps_list, dist_thrsh):
                tp += 1

        _recall = tp / len(gt_gps_list)
        _precision = tp / len(result_gps_list)

        return _recall, _precision
