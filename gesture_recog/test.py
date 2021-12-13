import os
import sys
os_path = os.path.join(os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")) , 'det_model')
sys.path.append(os.path.join(os_path, 'det_forward'))
print(os_path)
import det_pts_bbox
import cv2
import os
# os_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../")) , 'det_model')
#D:\xinghuo\gesturere_cognition\det_model\configs\exp_20210608_0_spark\eval_yolov3_v2m4.yaml
framework = 'yolov3'
config = os.path.join(os_path, 'configs/exp_20210608_0_spark/eval_yolov3_v2m4.yaml')
weights = os.path.join(os_path,'models/exp_20210608_0_spark/train_yolov3_V2M4_Spark_e150s0.pth')
data = os.path.join(os_path, 'mac_1.jpeg')
isCuda = False



model, num_heads, net_input_size, reductions, anchors, cls_loss_type, conf_thresh, num_class, nms_param = det_pts_bbox.det_init(framework, config, weights)

img = cv2.imread(data)

# test
im = det_pts_bbox.show_det(img, model, num_heads, net_input_size, reductions, anchors, cls_loss_type, conf_thresh, num_class, nms_param, data, isCuda=isCuda,stext='text')
cv2.imshow("video", im)
cv2.waitKey (0)
cv2.destroyAllWindows()